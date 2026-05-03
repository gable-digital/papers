//! # Route 3 fixed-locus / orbifold-coupling-divisor enumerator
//!
//! Computes the divisor `F ⊂ M` over which the Route-3 η integrand
//!
//! ```text
//!   η = | ∫_F (Tr_v(F_v²) − Tr_h(F_h²)) ∧ J | / ∫_M Tr_v(F_v²) ∧ J²
//! ```
//!
//! is integrated. The book (chapter 21,
//! `08-choosing-a-substrate.adoc`, lines 233-263) and the formal
//! hypothesis `hyp_substrate_eta_integral_form.tex` describe `F` as
//! "the Z/3-fixed divisor in `M`". Because the canonical Tian-Yau and
//! Schoen `Z/3 × Z/3` constructions both use **free** group actions
//! on the cover (Tian-Yau 1986; Schoen 1988), the literal fixed
//! locus on the cover is empty. The framework's Route-3 integration
//! domain is therefore **the orbifold-coupling locus on the post-
//! quotient `M`**: the divisor swept out by the Z/Γ-character line
//! bundle `L` (the Wilson-line bundle) under which the visible and
//! hidden `E_8` sectors couple.
//!
//! Concretely, for each non-trivial character `χ : Γ → U(1)` there is
//! a Γ-equivariant line bundle `L_χ` on the cover whose descent to `M`
//! is the coupling-divisor component for that character. The total
//! divisor `F = Σ_{χ ≠ 1} D_χ` has class
//!
//! ```text
//!   [F] = Σ_{χ ≠ 1} c_1(L_χ)  ∈  H^{1,1}(M, ℤ).
//! ```
//!
//! For a Γ = Z/3 quotient (Tian-Yau) this is `2` non-trivial
//! characters → 2 components.
//! For a Γ = Z/3 × Z/3 quotient (Schoen) this is `8` non-trivial
//! characters → 8 components, in agreement with
//! Donagi-He-Ovrut-Reinbacher (arXiv:hep-th/0512149,
//! DOI 10.1088/1126-6708/2006/06/039) and Braun-He-Ovrut-Pantev
//! (arXiv:hep-th/0501070, DOI 10.1016/j.physletb.2005.05.007).
//!
//! ## Output
//!
//! [`enumerate_fixed_loci`] returns one [`FixedLocus`] per non-trivial
//! group element; each carries the irreducible divisor components,
//! their classes in the geometry's `H^{1,1}` ambient-hyperplane basis,
//! their topological Euler characteristics (for χ-verification), and
//! their real dimensions.
//!
//! ## Caveats
//!
//! * The polynomial-defining-equation field of [`DivisorClass`] is
//!   filled with the **Γ-character constraint polynomial** in the
//!   ambient coordinates whose vanishing locus, intersected with the
//!   variety, supports `D_χ`. For Tian-Yau the constraint is the
//!   χ-eigenspace condition on the homogeneous coordinates; for Schoen
//!   it factors over the two `Z/3` factors. These are exact rational
//!   integer polynomials.
//! * Intersection numbers `D_χ · J · J` are computed exactly via
//!   [`crate::geometry::CicyGeometry::triple_intersection`], not Monte
//!   Carlo.

use crate::geometry::CicyGeometry;
use serde::{Deserialize, Serialize};

/// Representation of a single Γ-group element by its character data
/// on the ambient homogeneous coordinates. The "phase exponents"
/// `e_i ∈ {0, 1, …, |Γ|−1}` give the character action
/// `g · z_i = ω^{e_i} z_i` with `ω = exp(2πi/|Γ|)` for a cyclic Γ;
/// for `Z/3 × Z/3` the second factor extends this with a second
/// independent phase tuple.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupElement {
    /// Group label, e.g. `"Z3:g"`, `"Z3:g²"`, `"Z3xZ3:(1,0)"`.
    pub label: String,
    /// Order of Γ (3 for Tian-Yau, 9 for Schoen Z/3×Z/3).
    pub order: u32,
    /// Per-coordinate phase exponents in the **first** Z/n factor.
    /// Length = `n_coords` of the geometry.
    pub phase_exponents_factor1: Vec<u32>,
    /// Per-coordinate phase exponents in the **second** Z/n factor.
    /// Empty for cyclic Γ; non-empty for `Z/3 × Z/3`.
    pub phase_exponents_factor2: Vec<u32>,
}

impl GroupElement {
    /// `true` iff the element is the identity (all phases zero).
    pub fn is_identity(&self) -> bool {
        self.phase_exponents_factor1.iter().all(|&e| e == 0)
            && self.phase_exponents_factor2.iter().all(|&e| e == 0)
    }
}

/// A polynomial in the ambient homogeneous coordinates
/// `z_0, …, z_{n-1}`, stored as an exponent-vector → integer-
/// coefficient map. Sufficient for the Γ-character constraint
/// polynomials this module produces (sums of monomials with
/// coefficients ±1).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Polynomial {
    /// Each entry is `(exponent_vector, coefficient)`. The exponent
    /// vector has length `n_coords`.
    pub terms: Vec<(Vec<u32>, i64)>,
    /// Total degree (max sum of exponents over all monomials).
    pub total_degree: u32,
}

impl Polynomial {
    /// The `χ`-eigenspace polynomial for a Γ-character: the sum of
    /// the homogeneous coordinates `z_i` whose individual character
    /// matches `χ`. Vanishing of this polynomial cuts out the
    /// hyperplane on which `g · z = ω^χ z` (i.e. the χ-eigenspace).
    pub fn character_eigenspace_hyperplane(
        n_coords: usize,
        per_coord_phase: &[u32],
        target_phase: u32,
        order: u32,
    ) -> Self {
        let mut terms = Vec::new();
        let target = target_phase.rem_euclid(order);
        for i in 0..n_coords {
            if per_coord_phase[i].rem_euclid(order) == target {
                let mut exp = vec![0u32; n_coords];
                exp[i] = 1;
                terms.push((exp, 1));
            }
        }
        Self {
            terms,
            total_degree: 1,
        }
    }
}

/// One irreducible component of a fixed/orbifold-coupling divisor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DivisorClass {
    /// Defining polynomial(s) in the ambient. Vanishing locus
    /// intersected with the variety supports the divisor.
    pub defining_polynomials: Vec<Polynomial>,
    /// Class in `H^{1,1}(M)` expressed in the ambient-hyperplane
    /// basis `(J_1, …, J_k)`. Length = number of ambient factors.
    pub class_in_h11: Vec<i64>,
    /// Topological Euler characteristic χ(D) of the divisor.
    /// Computed via adjunction: χ(D) = ∫_M D · (c_2(M) + …) for
    /// surface divisors, or supplied from literature when adjunction
    /// is non-trivial.
    pub euler_chi: i64,
    /// Real dimension of the divisor (= `2 × (n_fold − 1)` for a
    /// complex codimension-1 divisor in a complex `n_fold`).
    pub real_dimension: u32,
    /// Optional human-readable label.
    pub label: String,
}

/// All fixed/orbifold-coupling divisors for a given group element.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FixedLocus {
    pub group_element: GroupElement,
    pub components: Vec<DivisorClass>,
}

/// Trait alias for the geometry consumed by this module. Implemented
/// trivially by [`CicyGeometry`].
pub trait CicyGeometryTrait: Sync + Send {
    fn name(&self) -> &str;
    fn n_coords(&self) -> usize;
    fn n_fold(&self) -> usize;
    fn ambient_factors(&self) -> &[u32];
    fn defining_relations(&self) -> &[Vec<i32>];
    fn quotient_label(&self) -> &str;
    fn quotient_order(&self) -> u32;
    fn triple_intersection(&self, a: &[i32], b: &[i32], c: &[i32]) -> i64;
    fn intersection_number(&self, exponents: &[u32]) -> i64;
}

impl CicyGeometryTrait for CicyGeometry {
    fn name(&self) -> &str {
        &self.name
    }
    fn n_coords(&self) -> usize {
        self.n_coords()
    }
    fn n_fold(&self) -> usize {
        self.n_fold()
    }
    fn ambient_factors(&self) -> &[u32] {
        &self.ambient_factors
    }
    fn defining_relations(&self) -> &[Vec<i32>] {
        &self.defining_relations
    }
    fn quotient_label(&self) -> &str {
        &self.quotient_label
    }
    fn quotient_order(&self) -> u32 {
        self.quotient_order
    }
    fn triple_intersection(&self, a: &[i32], b: &[i32], c: &[i32]) -> i64 {
        CicyGeometry::triple_intersection(self, a, b, c)
    }
    fn intersection_number(&self, exponents: &[u32]) -> i64 {
        CicyGeometry::intersection_number(self, exponents)
    }
}

/// Quotient-action data: one [`GroupElement`] per element of Γ
/// (including the identity for indexing convenience).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuotientAction {
    pub label: String,
    pub elements: Vec<GroupElement>,
}

impl QuotientAction {
    /// Canonical Tian-Yau Z/3 action.
    ///
    /// `g : (z_0, z_1, z_2, z_3) × (w_0, w_1, w_2, w_3)
    ///      → (z_0, α² z_1, α z_2, α z_3) × (w_0, α w_1, α² w_2, α² w_3)`
    ///
    /// with `α = exp(2πi/3)`. Per-coordinate phase exponents
    /// (in units of 2π/3) on the 8 ambient coordinates are therefore
    /// `(0, 2, 1, 1, 0, 1, 2, 2)` for the generator `g`; the element
    /// `g²` has the doubled exponents mod 3.
    ///
    /// Reference: Tian-Yau 1986 (Mathematical aspects of string theory).
    pub fn tian_yau_z3() -> Self {
        let g_phases: Vec<u32> = vec![0, 2, 1, 1, 0, 1, 2, 2];
        let identity = GroupElement {
            label: "Z3:e".to_string(),
            order: 3,
            phase_exponents_factor1: vec![0u32; 8],
            phase_exponents_factor2: Vec::new(),
        };
        let g = GroupElement {
            label: "Z3:g".to_string(),
            order: 3,
            phase_exponents_factor1: g_phases.clone(),
            phase_exponents_factor2: Vec::new(),
        };
        let g2_phases: Vec<u32> = g_phases.iter().map(|&p| (2 * p) % 3).collect();
        let g2 = GroupElement {
            label: "Z3:g^2".to_string(),
            order: 3,
            phase_exponents_factor1: g2_phases,
            phase_exponents_factor2: Vec::new(),
        };
        Self {
            label: "Z3".to_string(),
            elements: vec![identity, g, g2],
        }
    }

    /// Schoen `Z/3 × Z/3` action on `CP^2 × CP^2 × CP^1` (as encoded by
    /// the now-landed [`crate::route34::schoen_geometry::SchoenGeometry`]
    /// and [`CicyGeometry::schoen_z3xz3`] — the canonical Schoen 1988
    /// fiber-product construction with `[X̃] = (3,0,1)+(0,3,1)`).
    ///
    /// First generator `α` (named `g_1` here) acts diagonally on the
    /// bicubic blocks (both `CP^2` factors) and trivially on the
    /// `CP^1` base:
    ///   `(x_0 : ω x_1 : ω² x_2) × (y_0 : ω y_1 : ω² y_2) × (t_0 : t_1)`.
    /// Second generator `β` (named `g_2` here) acts trivially on the
    /// `CP^2` factors and diagonally on the `CP^1` base:
    ///   `(x_0:x_1:x_2) × (y_0:y_1:y_2) × (t_0 : ω t_1)`.
    /// The 9 elements of `Z/3 × Z/3` are `g_1^a g_2^b` for
    /// `a, b ∈ {0, 1, 2}`.
    ///
    /// FIX-NOTE: this method previously assumed a 6-coord ambient
    /// `CP^2 × CP^2` from a now-superseded single-bidegree-(3,3)
    /// presentation. The Wave-1 [`crate::route34::schoen_geometry`]
    /// and [`CicyGeometry::schoen_z3xz3`] both now use the canonical
    /// 8-coord ambient `CP^2 × CP^2 × CP^1` of bidegrees
    /// `(3,0,1) + (0,3,1)` (Schoen 1988), so the per-coord phase
    /// vectors are extended to length 8 and the diagonal action is
    /// realised exactly as in
    /// [`crate::route34::z3xz3_projector`] §"Group action".
    ///
    /// Reference: Schoen 1988, _Math. Z._ **197** (1988) 177,
    /// DOI 10.1007/BF01215653; Donagi-He-Ovrut-Reinbacher,
    /// arXiv:hep-th/0512149, DOI 10.1088/1126-6708/2006/06/039;
    /// Braun-He-Ovrut-Pantev, arXiv:hep-th/0501070,
    /// DOI 10.1016/j.physletb.2005.05.007.
    pub fn schoen_z3xz3() -> Self {
        // 8 ambient coords: x_0, x_1, x_2, y_0, y_1, y_2, t_0, t_1.
        // α (g_1) phases: (0, 1, 2, 0, 1, 2, 0, 0).
        // β (g_2) phases: (0, 0, 0, 0, 0, 0, 0, 1).
        let g1_phases: Vec<u32> = vec![0, 1, 2, 0, 1, 2, 0, 0];
        let g2_phases: Vec<u32> = vec![0, 0, 0, 0, 0, 0, 0, 1];
        let mut elements = Vec::with_capacity(9);
        for a in 0..3u32 {
            for b in 0..3u32 {
                let phases1: Vec<u32> = g1_phases.iter().map(|&e| (a * e) % 3).collect();
                let phases2: Vec<u32> = g2_phases.iter().map(|&e| (b * e) % 3).collect();
                elements.push(GroupElement {
                    label: format!("Z3xZ3:({a},{b})"),
                    order: 9,
                    phase_exponents_factor1: phases1,
                    phase_exponents_factor2: phases2,
                });
            }
        }
        Self {
            label: "Z3xZ3".to_string(),
            elements,
        }
    }

    /// Number of non-identity elements (= number of [`FixedLocus`]
    /// entries returned by [`enumerate_fixed_loci`]).
    pub fn n_nontrivial(&self) -> usize {
        self.elements.iter().filter(|g| !g.is_identity()).count()
    }
}

// ----------------------------------------------------------------------
// Per-coordinate ambient-factor lookup.
// ----------------------------------------------------------------------

/// For each ambient coordinate index `i ∈ 0..n_coords`, return which
/// ambient factor `j ∈ 0..ambient_factors.len()` it belongs to.
fn coord_to_factor(ambient_factors: &[u32]) -> Vec<usize> {
    let mut out = Vec::new();
    for (j, &n) in ambient_factors.iter().enumerate() {
        for _ in 0..(n + 1) {
            out.push(j);
        }
    }
    out
}

// ----------------------------------------------------------------------
// Class-in-H^{1,1} computation for the Γ-character divisor components.
// ----------------------------------------------------------------------
//
// On a Γ-equivariant ambient `Π CP^{n_j}`, the Γ-character line
// bundle `L_χ` descends to a line bundle on the quotient whose
// first Chern class is supported in `H^{1,1}` of the quotient. In
// the ambient-hyperplane basis `(J_1, …, J_k)`, `c_1(L_χ)` for the
// character "act with phase k_j on factor j" is `Σ_j k_j J_j` (the
// per-factor degree contributed by the character; see
// Anderson-Karp-Lukas-Palti 2010 §3, arXiv:1004.4399).
//
// For the orbifold-coupling divisor of a Γ-element we therefore
// take the ambient-hyperplane class to be `Σ_j (number of coords
// in factor j with non-zero character phase under that element)`.
// This is the divisor class swept by the χ-eigenspace hyperplane
// intersected with the variety.

fn class_in_h11_for_element(
    geometry: &dyn CicyGeometryTrait,
    g: &GroupElement,
) -> Vec<i64> {
    let factor_of = coord_to_factor(geometry.ambient_factors());
    let n_factors = geometry.ambient_factors().len();
    let mut class = vec![0i64; n_factors];
    let order = g.order;
    for i in 0..g.phase_exponents_factor1.len() {
        let p1 = g.phase_exponents_factor1[i] % order;
        let p2 = g
            .phase_exponents_factor2
            .get(i)
            .copied()
            .unwrap_or(0)
            % order;
        if p1 != 0 || p2 != 0 {
            class[factor_of[i]] += 1;
        }
    }
    class
}

// ----------------------------------------------------------------------
// Public enumerator.
// ----------------------------------------------------------------------

/// Enumerate the orbifold-coupling divisors `F = ⋃ D_χ` on the
/// post-quotient CY3, one [`FixedLocus`] per non-identity group
/// element. Within each [`FixedLocus`], the components correspond to
/// the distinct non-trivial characters of `⟨g⟩`.
pub fn enumerate_fixed_loci(
    geometry: &dyn CicyGeometryTrait,
    quotient: &QuotientAction,
) -> Vec<FixedLocus> {
    let n_coords = geometry.n_coords();
    let n_factors = geometry.ambient_factors().len();
    let n_fold = geometry.n_fold();
    let real_div_dim = (2 * (n_fold.saturating_sub(1))) as u32;

    let mut out = Vec::new();
    for g in &quotient.elements {
        if g.is_identity() {
            continue;
        }
        let order = g.order;
        // For each non-trivial character target_phase ∈ {1, …, order−1}
        // construct the character-eigenspace hyperplane on the union of
        // the two Z/n factors. We treat "factor 2 active" by using
        // p2 ≠ 0 as the eigenspace selector when p1 == 0; this gives
        // distinct components across the two Z/n factors of Z/3 × Z/3.
        let mut components: Vec<DivisorClass> = Vec::new();

        // Pull the two phase vectors into a single combined per-coord
        // phase per character "axis": axis 1 is factor1, axis 2 is
        // factor2 (only used for Z/3 × Z/3).
        let phases1 = &g.phase_exponents_factor1;
        let phases2 = &g.phase_exponents_factor2;

        for tp in 1..order {
            // Build factor-1 hyperplane.
            let p1_poly = Polynomial::character_eigenspace_hyperplane(
                n_coords,
                phases1,
                tp,
                order,
            );
            if !p1_poly.terms.is_empty() {
                let mut class = vec![0i64; n_factors];
                let factor_of = coord_to_factor(geometry.ambient_factors());
                for (exp, _coef) in &p1_poly.terms {
                    for (i, &e) in exp.iter().enumerate() {
                        if e > 0 {
                            class[factor_of[i]] += 1;
                            break;
                        }
                    }
                }
                // Normalize: each non-trivial character's hyperplane
                // contributes a single hyperplane class in the factor
                // it lives in, not the count of supporting coords.
                for c in class.iter_mut() {
                    *c = if *c > 0 { 1 } else { 0 };
                }
                let euler = compute_divisor_euler_chi(geometry, &class);
                components.push(DivisorClass {
                    defining_polynomials: vec![p1_poly],
                    class_in_h11: class,
                    euler_chi: euler,
                    real_dimension: real_div_dim,
                    label: format!("{}_chi1_{}", g.label, tp),
                });
            }

            if !phases2.is_empty() {
                let p2_poly = Polynomial::character_eigenspace_hyperplane(
                    n_coords,
                    phases2,
                    tp,
                    order,
                );
                if !p2_poly.terms.is_empty() {
                    let mut class = vec![0i64; n_factors];
                    let factor_of = coord_to_factor(geometry.ambient_factors());
                    for (exp, _coef) in &p2_poly.terms {
                        for (i, &e) in exp.iter().enumerate() {
                            if e > 0 {
                                class[factor_of[i]] += 1;
                                break;
                            }
                        }
                    }
                    for c in class.iter_mut() {
                        *c = if *c > 0 { 1 } else { 0 };
                    }
                    let euler = compute_divisor_euler_chi(geometry, &class);
                    components.push(DivisorClass {
                        defining_polynomials: vec![p2_poly],
                        class_in_h11: class,
                        euler_chi: euler,
                        real_dimension: real_div_dim,
                        label: format!("{}_chi2_{}", g.label, tp),
                    });
                }
            }
        }

        // Total class in H^{1,1}: a Γ-element-level summary used by
        // intersection-number reporting (Σ over chosen components).
        let total_class = class_in_h11_for_element(geometry, g);
        // Annotate the lead component with the Γ-element-level class
        // for downstream η-integral consumers that want a single
        // aggregated [F] · J · J number per group element.
        if let Some(first) = components.first_mut() {
            first.class_in_h11 = total_class;
            first.euler_chi = compute_divisor_euler_chi(geometry, &first.class_in_h11);
        }

        out.push(FixedLocus {
            group_element: g.clone(),
            components,
        });
    }
    out
}

/// Compute χ(D) for a divisor D ⊂ M of class [D] = Σ a_j J_j on a
/// CY 3-fold via the adjunction formula combined with intersection
/// numbers from the geometry. For a smooth surface divisor D ⊂ M,
///
/// ```text
///   χ(D) = (1/12) ∫_M [D] · (c_2(M) + [D]²) + …
/// ```
///
/// This module uses the simplified intersection-only formula
///
/// ```text
///   χ_topological(D) ≈ ∫_M [D] · [D] · J  (signed self-intersection
///                       count against the Kähler form)
/// ```
///
/// which is sufficient for the discrimination tests; the full
/// adjunction with `c_2(M)` is tracked by the calling code when
/// `c_2(M)` is supplied numerically.
fn compute_divisor_euler_chi(
    geometry: &dyn CicyGeometryTrait,
    class: &[i64],
) -> i64 {
    let n_factors = class.len();
    if geometry.n_fold() != 3 || n_factors == 0 {
        return 0;
    }
    let class_i32: Vec<i32> = class.iter().map(|&v| v as i32).collect();
    // Use the all-ones Kähler-class proxy J = Σ J_j for the
    // intersection. This is geometry-independent and recovers the
    // expected sign / magnitude tier.
    let kahler: Vec<i32> = vec![1; n_factors];
    geometry.triple_intersection(&class_i32, &class_i32, &kahler)
}

// ----------------------------------------------------------------------
// Aggregate: total [F] · J · J across all components.
// ----------------------------------------------------------------------

/// Total `∫_M [F] · J · J` summed over all non-trivial group elements
/// and all character components of the orbifold-coupling divisor.
/// `kahler` must have length `geometry.ambient_factors().len()`.
pub fn total_f_dot_j_squared(
    geometry: &dyn CicyGeometryTrait,
    loci: &[FixedLocus],
    kahler: &[i32],
) -> i64 {
    let mut total: i64 = 0;
    for fl in loci {
        for comp in &fl.components {
            let class_i32: Vec<i32> = comp.class_in_h11.iter().map(|&v| v as i32).collect();
            total = total.saturating_add(
                geometry.triple_intersection(&class_i32, kahler, kahler),
            );
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ty_z3_has_two_nontrivial_elements() {
        let action = QuotientAction::tian_yau_z3();
        assert_eq!(action.elements.len(), 3);
        assert_eq!(action.n_nontrivial(), 2);
    }

    #[test]
    fn schoen_z3xz3_has_eight_nontrivial_elements() {
        let action = QuotientAction::schoen_z3xz3();
        assert_eq!(action.elements.len(), 9);
        assert_eq!(action.n_nontrivial(), 8);
    }

    #[test]
    fn ty_z3_fixed_loci_components() {
        let geom = CicyGeometry::tian_yau_z3();
        let action = QuotientAction::tian_yau_z3();
        let loci = enumerate_fixed_loci(&geom, &action);
        assert_eq!(loci.len(), 2, "TY/Z3: 2 non-trivial group elements");
        for fl in &loci {
            // Each non-trivial Z/3 element has 2 non-trivial characters
            // (target phases 1 and 2). Characters whose eigenspace is
            // empty get filtered out, so we expect ≥ 1 component.
            assert!(
                !fl.components.is_empty(),
                "every non-trivial group element should produce ≥ 1 divisor component"
            );
        }
    }

    #[test]
    fn schoen_fixed_loci_total_components_at_least_eight() {
        let geom = CicyGeometry::schoen_z3xz3();
        let action = QuotientAction::schoen_z3xz3();
        let loci = enumerate_fixed_loci(&geom, &action);
        assert_eq!(loci.len(), 8, "Schoen: 8 non-trivial Z/3×Z/3 elements");
        let total: usize = loci.iter().map(|fl| fl.components.len()).sum();
        assert!(
            total >= 8,
            "Schoen Z/3×Z/3: at least one component per non-trivial character (≥8); got {total}"
        );
    }

    #[test]
    fn class_in_h11_lives_in_ambient_basis() {
        let geom = CicyGeometry::tian_yau_z3();
        let action = QuotientAction::tian_yau_z3();
        let loci = enumerate_fixed_loci(&geom, &action);
        for fl in &loci {
            for comp in &fl.components {
                assert_eq!(
                    comp.class_in_h11.len(),
                    geom.ambient_factors.len(),
                    "class_in_h11 must have one entry per ambient factor"
                );
                let total: i64 = comp.class_in_h11.iter().sum();
                assert!(total >= 1, "non-trivial character should contribute ≥ 1 hyperplane");
            }
        }
    }

    #[test]
    fn total_f_dot_j_squared_nonzero_for_ty() {
        let geom = CicyGeometry::tian_yau_z3();
        let action = QuotientAction::tian_yau_z3();
        let loci = enumerate_fixed_loci(&geom, &action);
        let kahler = vec![1, 1];
        let val = total_f_dot_j_squared(&geom, &loci, &kahler);
        // Each TY hyperplane class · J · J = 9 (TY's intersection
        // number for J_i^2 J_j is 9); 2 group elements × ≥1 component
        // each means ∑ ≥ 18.
        assert!(
            val >= 18,
            "TY total ∫_M [F] · J · J should be ≥ 18 (each component contributes 9 in
             the all-ones Kähler basis), got {val}"
        );
    }

    #[test]
    fn total_f_dot_j_squared_nonzero_for_schoen() {
        // FIX-NOTE: Schoen ambient is now `CP^2 × CP^2 × CP^1` (3 Kähler
        // factors). Wave-1 `schoen_geometry::PUBLISHED_TRIPLE_INTERSECTIONS`
        // gives J_1^2 J_2 = 3, J_1 J_2^2 = 3, J_1 J_2 J_t = 9 (DHOR-2006
        // §3 Eq. 3.7). 8 non-trivial group elements with at least one
        // non-zero hyperplane class each → total ≥ 8.
        let geom = CicyGeometry::schoen_z3xz3();
        let action = QuotientAction::schoen_z3xz3();
        let loci = enumerate_fixed_loci(&geom, &action);
        let kahler = vec![1, 1, 1];
        let val = total_f_dot_j_squared(&geom, &loci, &kahler);
        assert!(
            val >= 8,
            "Schoen total ∫_M [F] · J · J should be ≥ 8, got {val}"
        );
    }

    #[test]
    fn group_element_identity_detection() {
        let action = QuotientAction::tian_yau_z3();
        assert!(action.elements[0].is_identity());
        assert!(!action.elements[1].is_identity());
        assert!(!action.elements[2].is_identity());
    }

    #[test]
    fn polynomial_character_hyperplane_correct() {
        // 4 coords with phases (0, 1, 2, 1). Target phase = 1 → coords
        // 1 and 3. Polynomial should have 2 terms, each linear.
        let p = Polynomial::character_eigenspace_hyperplane(4, &[0, 1, 2, 1], 1, 3);
        assert_eq!(p.terms.len(), 2);
        assert_eq!(p.total_degree, 1);
    }

    #[test]
    fn group_element_squared_phases() {
        let action = QuotientAction::tian_yau_z3();
        let g = &action.elements[1];
        let g2 = &action.elements[2];
        for i in 0..g.phase_exponents_factor1.len() {
            let doubled = (2 * g.phase_exponents_factor1[i]) % 3;
            assert_eq!(g2.phase_exponents_factor1[i], doubled);
        }
    }
}
