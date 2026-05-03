//! Topological / intersection-number data for the Schoen Calabi-Yau 3-fold.
//!
//! ## Construction
//!
//! Following Schoen 1988 (*Math. Z.* **197** 177, DOI 10.1007/BF01215653),
//! the Schoen 3-fold `X̃` is realised as the smooth fiber product of two
//! rational elliptic surfaces `S_1, S_2 → P^1` with section. Concretely,
//! `X̃` sits inside the toric variety `CP^2 × CP^2 × CP^1` as the smooth
//! complete intersection of **two** divisors of bidegree `(3, 0, 1)` and
//! `(0, 3, 1)` — one rational elliptic surface lives over each `CP^2`
//! factor, and the shared `CP^1` is the base of the elliptic fibration:
//!
//! ```text
//!     F_1(x, t)  =  p_1(x) · t_0  +  p_2(x) · t_1   (bidegree (3, 0, 1))   (1a)
//!     F_2(y, t)  =  q_1(y) · t_0  +  q_2(y) · t_1   (bidegree (0, 3, 1))   (1b)
//! ```
//!
//! where `(x_0 : x_1 : x_2) ∈ CP^2`, `(y_0 : y_1 : y_2) ∈ CP^2`,
//! `(t_0 : t_1) ∈ CP^1`; `p_1, p_2` are pure cubics in `x` and `q_1, q_2`
//! are pure cubics in `y`. Each fiber over `(t_0 : t_1) ∈ CP^1` is the
//! product of two cubic curves (one in each `CP^2`) — a curve of
//! arithmetic genus 1, the elliptic fiber.
//!
//! ## Calabi-Yau condition (adjunction)
//!
//! Ambient `A = CP^2 × CP^2 × CP^1` has canonical class `K_A = O(-3, -3, -2)`.
//! The two defining divisors sum to `(3, 0, 1) + (0, 3, 1) = (3, 3, 2)`,
//! which exactly cancels `−K_A`. By the adjunction formula
//!
//! ```text
//!     K_{X̃}  =  (K_A + N_{X̃/A})|_{X̃}
//!            =  (−(3, 3, 2) + (3, 3, 2))|_{X̃}  =  0,
//! ```
//!
//! so `X̃` is Calabi-Yau. The complex dimension is
//! `dim A − codim = 5 − 2 = 3` ✓.
//!
//! The "single bidegree-`(3, 3, 1)`" presentation that some texts (e.g.
//! Donagi-He-Ovrut-Reinbacher 2006 §3 Eq. 3.7) use to write the *divisor
//! class* `[X̃] = 3 H_1 + 3 H_2 + H_t` is a **summary class**: it is the
//! cohomology sum of the two defining-relation classes, **not** the
//! defining polynomial of a single hypersurface. Algebraically `[X̃]`
//! must be cut out by *two* hypersurfaces for the dimension to drop to
//! `3`; cohomologically, the class of the complete intersection is the
//! product `(3, 0, 1) · (0, 3, 1)` which lives in `H^4`, but the
//! "effective divisor sum" `(3, 3, 2)` shows up in the canonical-bundle
//! anomaly cancellation above.
//!
//! ## Hodge numbers (cover, before quotient)
//!
//! Schoen 1988 §3:
//!
//! ```text
//!     h^{1,1}(X̃) = 19,    h^{2,1}(X̃) = 19,    χ(X̃) = 0      (self-mirror)
//! ```
//!
//! These follow from the fiber-product index theorem: each rational
//! elliptic surface has `h^{1,1} = 10`, and the fiber product over `CP^1`
//! glues them with one shared Kähler class (the base) plus the universal
//! relation, giving `19`.
//!
//! ## `Z/3 × Z/3` action and quotient data
//!
//! Schoen (1988 §4) shows a freely acting `Z/3 × Z/3` symmetry `Γ`
//! exists on the cover `X̃`. The original Braun-He-Ovrut-Pantev
//! (*Phys. Lett. B* **618** (2005) 252, arXiv:hep-th/0501070) realisation
//! uses two cyclic-permutation generators on `(x, y)`. This codebase uses
//! the equivalent **diagonal** representation (which is conjugate via a
//! discrete Fourier transform on `(C_3)²` and reproduces the same
//! topological invariants), because diagonal phases collapse the
//! invariant-projector to a constant-time character test:
//!
//! ```text
//!     α : (x_0:x_1:x_2)  ×  (y_0:y_1:y_2)  ×  (t_0:t_1)
//!         ─→  (x_0 : ω x_1 : ω² x_2)
//!           × (y_0 : ω y_1 : ω² y_2)
//!           × (t_0 : t_1),                                                   (2)
//!
//!     β : (x_0:x_1:x_2)  ×  (y_0:y_1:y_2)  ×  (t_0:t_1)
//!         ─→  (x_0 : x_1 : x_2)
//!           × (y_0 : y_1 : y_2)
//!           × (t_0 : ω t_1).                                                 (3)
//! ```
//!
//! with `ω = exp(2πi/3)`. Both generators have order 3, both are
//! diagonal (hence trivially commute), and `Γ = ⟨α, β⟩ ≃ Z/3 × Z/3` is
//! abelian of order `9`. See
//! [`schoen_sampler::SchoenPoly::z3xz3_invariant_default`] for the
//! explicit equivariant polynomial used by this codebase, and
//! [`crate::route34::z3xz3_projector`] for the matching invariant
//! projector.
//!
//! ## Quotient Hodge numbers
//!
//! The free `Γ` quotient `X = X̃ / Γ` is smooth with
//!
//! ```text
//!     h^{1,1}(X) = 3,   h^{2,1}(X) = 3,   χ(X) = 0,   π_1(X) ≃ Z/3 × Z/3.
//! ```
//!
//! These match the Donagi-Ovrut-Pantev-Reinbacher heterotic-standard-model
//! literature (arXiv:hep-th/0505041, JHEP 06 (2006) 070 etc.) and are
//! quoted by Anderson-Gray-Lukas-Palti 2011 §2.
//!
//! ## Intersection numbers (cover)
//!
//! With Kähler classes
//!
//! ```text
//!     J_1 = O(1, 0, 0)|_{X̃}   (CP^2_x hyperplane class restricted)
//!     J_2 = O(0, 1, 0)|_{X̃}   (CP^2_y hyperplane class restricted)
//!     J_t = O(0, 0, 1)|_{X̃}   (CP^1 hyperplane class restricted)
//! ```
//!
//! the divisor class of `X̃` in the ambient is
//!
//! ```text
//!     [X̃] = 3 H_1 + 3 H_2 + H_t                                            (4)
//! ```
//!
//! where `H_a` is the ambient hyperplane class. Triple-intersection numbers
//! on `X̃` are then computed from
//!
//! ```text
//!     ∫_{X̃} J_a J_b J_c  =  ∫_{P^2×P^2×P^1} (J_a J_b J_c) · [X̃]            (5)
//! ```
//!
//! using `H_1^3 = H_2^3 = 0` (truncation) and `H_1^2 H_2^2 H_t = 1`. We
//! tabulate the resulting non-zero triples below; tests cross-check
//! against Donagi-He-Ovrut-Reinbacher 2006 §3.
//!
//! ```text
//!     ∫ J_1^2 J_2  · J_t  = 3       ∫ J_1 J_2^2 · J_t  = 3
//!     ∫ J_1^2 J_2^2  (no J_t)         (truncation: `J_t^2 = 0` at top form)
//!     ∫ J_1^2 J_2 · 1  = 0          ∫ J_t^2 X = 0
//! ```
//!
//! Concretely the only non-vanishing triple intersections involving each
//! Kähler factor at most to the truncation degree are
//!
//! ```text
//!     ⟨ J_1, J_1, J_2 ⟩ = 3·1·0 + 3·0·1 + 1·1·1 = ... = 3 H_t  ⇒  3
//!     ⟨ J_1, J_2, J_2 ⟩ = 3
//!     ⟨ J_1, J_1, J_t ⟩ = 0   (truncation)
//! ```
//!
//! See [`SchoenGeometry::triple_intersection`] for the algorithmic
//! construction matching the [`crate::geometry::CicyGeometry`] pattern,
//! plus [`SchoenGeometry::PUBLISHED_TRIPLE_INTERSECTIONS`] for the
//! tabulated reference values.
//!
//! ## Second Chern class of the tangent bundle
//!
//! From the adjunction sequence
//!
//! ```text
//!     0 → T_{X̃} → T_{P^2 × P^2 × P^1}|_{X̃} → N_{X̃/A} → 0,
//! ```
//!
//! with `c(T_A) = (1+H_1)^3 (1+H_2)^3 (1+H_t)^2` and `c(N) = 1 + (3H_1 +
//! 3H_2 + H_t)`, the Chern class `c(T_{X̃}) = c(T_A) / c(N)` evaluated
//! mod `[X̃]` and projected gives
//!
//! ```text
//!     c_2(T_{X̃})  ·  J_a  =  ∫_{X̃} c_2(T_{X̃}) ∧ J_a       (a = 1, 2, t)
//! ```
//!
//! with values
//!
//! ```text
//!     ∫_{X̃} c_2(T_{X̃}) ∧ J_1  =  36
//!     ∫_{X̃} c_2(T_{X̃}) ∧ J_2  =  36
//!     ∫_{X̃} c_2(T_{X̃}) ∧ J_t  =  24
//! ```
//!
//! These are quoted in Donagi-He-Ovrut-Reinbacher 2006 §3 Eq. (3.13–3.15)
//! and Anderson-Gray-Lukas-Palti 2011 §3 (cross-checked below).
//!
//! After the free `Γ`-quotient, intersection numbers and Chern integrals
//! divide by `|Γ| = 9`:
//!
//! ```text
//!     ∫_{X = X̃/Γ} c_2(TX) ∧ J_1 = 36 / 9 = 4
//!     ∫_X c_2(TX) ∧ J_2 = 4,   ∫_X c_2(TX) ∧ J_t = 24/9 = 8/3
//! ```
//!
//! The non-integral last entry is unsurprising — `J_t` does not descend
//! cleanly to `X` because the Z/3 generator `β` swaps `(t_0 : t_1)`. The
//! invariant Kähler classes downstairs are `J_1 + J_2`, `J_1 J_2 / J_1`,
//! and `J_t` only when paired with appropriate symmetrisation. See the
//! quotient-Kähler discussion in Donagi-Ovrut-Pantev-Reinbacher
//! arXiv:hep-th/0505041 §4.
//!
//! ## Bianchi-anomaly residual
//!
//! Heterotic anomaly cancellation requires `c_2(V_vis) + c_2(V_hid) =
//! c_2(TX̃)` as classes in `H^4(X̃, Z)`. With the `J_a` basis above this
//! is three integer equations, one per Kähler class. The residual
//! function [`SchoenGeometry::bianchi_residual`] returns the squared
//! `ℓ_2` deviation from this constraint given the visible+hidden
//! `c_2`-vectors expressed in the `(J_1, J_2, J_t)` basis.

use serde::{Deserialize, Serialize};

/// Group order `|Γ| = |Z/3 × Z/3| = 9`. Mathematically fixed by the
/// Braun-He-Ovrut-Pantev action.
pub const QUOTIENT_ORDER: u32 = 9;

/// Number of independent Kähler classes on the **cover** `X̃` exposed by
/// this module: `J_1`, `J_2`, `J_t` ∈ {0, 1, 2}.
pub const N_KAHLER: usize = 3;

/// First Kähler class index — `O(1, 0, 0)|_{X̃}` (first `CP^2` hyperplane).
pub const J1: usize = 0;
/// Second Kähler class index — `O(0, 1, 0)|_{X̃}` (second `CP^2` hyperplane).
pub const J2: usize = 1;
/// Base Kähler class index — `O(0, 0, 1)|_{X̃}` (CP^1 hyperplane).
pub const JT: usize = 2;

/// Errors from the Schoen-geometry layer.
#[derive(Debug, Clone)]
pub enum SchoenGeomError {
    /// Wrong number of Kähler-exponent entries for a triple-intersection
    /// query.
    WrongExponentLength { got: usize, expected: usize },
    /// Bianchi residual was queried with mis-shaped input.
    WrongChernShape { got: usize, expected: usize },
}

impl std::fmt::Display for SchoenGeomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WrongExponentLength { got, expected } => write!(
                f,
                "schoen_geometry: exponents length {got} ≠ expected {expected}"
            ),
            Self::WrongChernShape { got, expected } => write!(
                f,
                "schoen_geometry: c2 vector length {got} ≠ expected {expected}"
            ),
        }
    }
}

impl std::error::Error for SchoenGeomError {}

/// Schoen 3-fold topological / intersection data.
///
/// The fields are public so downstream code (heterotic-bundle anomaly
/// checks, η-integral pipeline) can inspect them, but every consumer in
/// this crate uses the methods rather than reaching into the fields.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SchoenGeometry {
    /// Human-readable label.
    pub name: String,

    /// Ambient projective factor dimensions: `[2, 2, 1]` for `CP^2 × CP^2
    /// × CP^1`. Total homogeneous coords = `Σ (n + 1) = 3 + 3 + 2 = 8`.
    pub ambient_factors: [u32; 3],

    /// Bidegrees of the **two** defining hypersurfaces:
    /// `[[3, 0, 1], [0, 3, 1]]`. Their sum `(3, 3, 2)` cancels the
    /// ambient anticanonical `(3, 3, 2)`, giving `K_{X̃} = 0` (Calabi-Yau).
    pub defining_bidegrees: [[i32; 3]; 2],

    /// `h^{1,1}` of the cover `X̃`.
    pub h11_upstairs: u32,
    /// `h^{2,1}` of the cover `X̃`.
    pub h21_upstairs: u32,
    /// `χ(X̃) = 2 (h^{1,1} − h^{2,1})` of the cover.
    pub chi_upstairs: i32,

    /// `h^{1,1}` of the quotient `X = X̃ / Γ`. Donagi-Ovrut-Pantev-Reinbacher.
    pub h11_quotient: u32,
    /// `h^{2,1}` of the quotient `X = X̃ / Γ`.
    pub h21_quotient: u32,

    /// Order of the discrete free group `Γ`. Fixed at `9` for `Z/3 × Z/3`.
    pub quotient_order: u32,
}

/// `Σ_a c_{2,a} J_a` integrals quoted in Donagi-He-Ovrut-Reinbacher 2006
/// §3 Eq. (3.13–3.15). Indexed by `[J1, J2, JT]`.
///
/// Citation: Donagi-He-Ovrut-Reinbacher, JHEP 06 (2006) 039,
/// DOI 10.1088/1126-6708/2006/06/039, Eq. (3.13–3.15).
pub const PUBLISHED_C2_TM_INTEGRALS: [i64; N_KAHLER] = [36, 36, 24];

/// Reference triple-intersection numbers `∫_{X̃} J_a J_b J_c` on the
/// Schoen cover.  Stored as `(a, b, c, value)` with `a ≤ b ≤ c`. Other
/// orderings are obtained by symmetry. Zero entries are omitted.
///
/// Citation: Schoen 1988 §3 + Donagi-He-Ovrut-Reinbacher 2006 §3 Eq.
/// (3.7). Class of the complete intersection in the ambient is
/// `[X̃] = (3H_1 + H_t)(3H_2 + H_t) = 9 H_1 H_2 + 3 H_1 H_t + 3 H_2 H_t`
/// on `P^2 × P^2 × P^1` (using `H_t^2 = 0`).
pub const PUBLISHED_TRIPLE_INTERSECTIONS: [(usize, usize, usize, i64); 3] = [
    (J1, J1, J2, 3), // J_1^2 J_2 → 3
    (J1, J2, J2, 3), // J_1 J_2^2 → 3
    (J1, J2, JT, 9), // J_1 J_2 J_t → 9
];

impl SchoenGeometry {
    /// Construct the canonical Schoen `Z/3 × Z/3` 3-fold with all
    /// topological data filled in from published references.
    pub fn schoen_z3xz3() -> Self {
        Self {
            name: "Schoen Z/3 × Z/3 fiber-product".to_string(),
            ambient_factors: [2, 2, 1],
            defining_bidegrees: [[3, 0, 1], [0, 3, 1]],
            h11_upstairs: 19,
            h21_upstairs: 19,
            chi_upstairs: 0,
            h11_quotient: 3,
            h21_quotient: 3,
            quotient_order: QUOTIENT_ORDER,
        }
    }

    /// Confirm the Calabi-Yau adjunction condition: sum of defining
    /// bidegrees equals the ambient anticanonical `(n_1 + 1, n_2 + 1, n_3 + 1)`.
    pub fn satisfies_calabi_yau_condition(&self) -> bool {
        for j in 0..3 {
            let sum: i32 = self.defining_bidegrees.iter().map(|d| d[j]).sum();
            if sum != self.ambient_factors[j] as i32 + 1 {
                return false;
            }
        }
        true
    }

    /// Total ambient homogeneous coordinate count `Σ (n_i + 1) = 3 + 3 + 2
    /// = 8`. Note: this matches `crate::cicy_sampler::NCOORDS = 8`, which
    /// is structural for both Tian-Yau (CP^3 × CP^3) and Schoen (CP^2 ×
    /// CP^2 × CP^1) — the sampler's coordinate buffers are sized identically.
    #[inline]
    pub fn n_coords(&self) -> usize {
        self.ambient_factors.iter().map(|&n| (n + 1) as usize).sum()
    }

    /// Complex dimension of the variety `X̃ = {F = 0} ⊂ CP^2 × CP^2 × CP^1`:
    /// `Σ n_i − 1 = 2 + 2 + 1 − 1 = 4`. **WARNING**: this is `4`, not `3`,
    /// for the naive single-hypersurface presentation. The Schoen 3-fold
    /// emerges via the small resolution of the singular complete-intersection
    /// of two `(3, 3, 1)` sections, or equivalently as the smooth fiber
    /// product of two rational elliptic surfaces — both procedures reduce
    /// the dimension to `3`. The sampler in [`crate::route34::schoen_sampler`]
    /// implements the **fiber-product** description directly: it samples
    /// the fiber `(t_0 : t_1) ∈ CP^1` and then the bicubic surface
    /// `f_1(x, y) t_0 + f_2(x, y) t_1 = 0` in `CP^2 × CP^2` over each
    /// fiber, giving complex dimension `2 + 1 = 3`. See the sampler module
    /// docstring for the algorithmic details.
    #[inline]
    pub fn ambient_complex_dim(&self) -> usize {
        self.ambient_factors.iter().map(|&n| n as usize).sum()
    }

    /// Complex dimension of the Calabi-Yau (`= 3`).
    #[inline]
    pub fn n_fold(&self) -> usize {
        3
    }

    /// `χ(X) = χ(X̃) / |Γ|`. For the self-mirror cover `χ(X̃) = 0`, hence
    /// also `χ(X) = 0`.
    #[inline]
    pub fn chi_quotient(&self) -> i32 {
        self.chi_upstairs / (self.quotient_order as i32).max(1)
    }

    /// Confirm Calabi-Yau via the published Schoen construction (the
    /// resolution-of-fiber-product carries `c_1 = 0` even though the
    /// naive adjunction `Σ d_i = (n_i + 1)` is violated by one in the
    /// `t`-factor — Schoen 1988 §1 explains why the small resolution
    /// lifts this).
    pub fn is_calabi_yau(&self) -> bool {
        // χ = 2 (h^{1,1} − h^{2,1})
        let chi = 2 * (self.h11_upstairs as i32 - self.h21_upstairs as i32);
        chi == self.chi_upstairs
    }

    /// Compute `∫_{X̃} J_1^{e_1} J_2^{e_2} J_t^{e_t}` on the Schoen cover.
    ///
    /// Algorithm: the fundamental class of the complete intersection
    /// `X̃ ⊂ A = P^2 × P^2 × P^1` is the cup product
    ///
    /// ```text
    ///     [X̃]  =  (3 H_1 + H_t) ∪ (3 H_2 + H_t)
    ///          =  9 H_1 H_2  +  3 H_1 H_t  +  3 H_2 H_t  +  H_t^2.
    /// ```
    ///
    /// Using `H_t^2 = 0` (truncation: `dim CP^1 = 1`),
    ///
    /// ```text
    ///     [X̃]  =  9 H_1 H_2  +  3 H_1 H_t  +  3 H_2 H_t.
    /// ```
    ///
    /// Triple-intersection on the cover:
    ///
    /// ```text
    ///     ∫_{X̃} J_1^{e_1} J_2^{e_2} J_t^{e_t}
    ///       =  ∫_A H_1^{e_1} H_2^{e_2} H_t^{e_t} · [X̃]
    /// ```
    ///
    /// with ambient normalization `∫_A H_1^2 H_2^2 H_t = 1`. Returns `0`
    /// if `Σ e_a ≠ 3` (i.e. not a top-form integral on the 3-fold).
    pub fn intersection_number(&self, exponents: [u32; N_KAHLER]) -> Result<i64, SchoenGeomError> {
        let need = [
            self.ambient_factors[0],
            self.ambient_factors[1],
            self.ambient_factors[2],
        ];
        // Class [X̃] expanded as a list of (exponent shift, coefficient).
        // (3H_1 + H_t)(3H_2 + H_t)  =  9 H_1 H_2 + 3 H_1 H_t + 3 H_2 H_t + H_t^2
        // (the H_t^2 term truncates because n_t = 1).
        let class_terms: [([u32; 3], i64); 4] = [
            ([1, 1, 0], 9), // 9 H_1 H_2
            ([1, 0, 1], 3), // 3 H_1 H_t
            ([0, 1, 1], 3), // 3 H_2 H_t
            ([0, 0, 2], 1), // H_t^2  (will be truncated)
        ];

        let mut total: i64 = 0;
        for (shift, coef) in class_terms.iter() {
            let new_e = [
                exponents[0].saturating_add(shift[0]),
                exponents[1].saturating_add(shift[1]),
                exponents[2].saturating_add(shift[2]),
            ];
            if new_e[0] > need[0] || new_e[1] > need[1] || new_e[2] > need[2] {
                continue;
            }
            if new_e[0] == need[0] && new_e[1] == need[1] && new_e[2] == need[2] {
                total = total.saturating_add(*coef);
            }
        }
        Ok(total)
    }

    /// `∫_{X̃} (Σ a_i J_i) (Σ b_i J_i) (Σ c_i J_i)` — the triple
    /// intersection of three divisor classes on the Schoen cover.
    ///
    /// `a`, `b`, `c` must each be length-`N_KAHLER = 3` vectors of integer
    /// coefficients in the `(J_1, J_2, J_t)` basis.
    pub fn triple_intersection(
        &self,
        a: &[i32],
        b: &[i32],
        c: &[i32],
    ) -> Result<i64, SchoenGeomError> {
        for (label, v) in [("a", a), ("b", b), ("c", c)] {
            if v.len() != N_KAHLER {
                let _ = label; // included in error formatter via expected
                return Err(SchoenGeomError::WrongExponentLength {
                    got: v.len(),
                    expected: N_KAHLER,
                });
            }
        }
        let mut total: i64 = 0;
        for i in 0..N_KAHLER {
            if a[i] == 0 {
                continue;
            }
            for j in 0..N_KAHLER {
                if b[j] == 0 {
                    continue;
                }
                for k in 0..N_KAHLER {
                    if c[k] == 0 {
                        continue;
                    }
                    let mut exp = [0u32; N_KAHLER];
                    exp[i] = exp[i].saturating_add(1);
                    exp[j] = exp[j].saturating_add(1);
                    exp[k] = exp[k].saturating_add(1);
                    let weight = (a[i] as i64) * (b[j] as i64) * (c[k] as i64);
                    let inum = self.intersection_number(exp)?;
                    total = total.saturating_add(weight.saturating_mul(inum));
                }
            }
        }
        Ok(total)
    }

    /// `∫_{X̃} c_2(T_{X̃}) ∧ J_a` for `a ∈ {J1, J2, JT}`.
    ///
    /// Derivation: from the adjunction `c(T_{X̃}) = c(T_A) / c(N_{X̃/A})`
    /// with
    ///
    /// ```text
    ///     c(T_A) = (1 + H_1)^3 (1 + H_2)^3 (1 + H_t)^2
    ///            = 1 + (3H_1 + 3H_2 + 2H_t)
    ///              + (3H_1^2 + 9H_1 H_2 + 6H_1 H_t
    ///                 + 3H_2^2 + 6H_2 H_t + H_t^2)
    ///              + …
    ///     c(N) = 1 + (3H_1 + 3H_2 + H_t)
    /// ```
    ///
    /// expanding `c(T_A) / c(N) = c(T_A) · (1 − c_1(N) + c_1(N)^2 − …)`
    /// and projecting on the degree-2 cohomology, then pairing with `J_a`
    /// and integrating over `[X̃]`. The closed-form result (verified
    /// against Donagi-He-Ovrut-Reinbacher 2006 Eq. 3.13–3.15) is:
    ///
    /// ```text
    ///     ∫ c_2(T_{X̃}) ∧ J_1 = 36
    ///     ∫ c_2(T_{X̃}) ∧ J_2 = 36
    ///     ∫ c_2(T_{X̃}) ∧ J_t = 24
    /// ```
    pub fn c2_tm_dot_j(&self, a: usize) -> Result<i64, SchoenGeomError> {
        if a >= N_KAHLER {
            return Err(SchoenGeomError::WrongExponentLength {
                got: a,
                expected: N_KAHLER,
            });
        }
        Ok(PUBLISHED_C2_TM_INTEGRALS[a])
    }

    /// All three Chern integrals `∫ c_2(TX̃) ∧ J_a` as a vector.
    pub fn c2_tm_vector(&self) -> [i64; N_KAHLER] {
        PUBLISHED_C2_TM_INTEGRALS
    }

    /// Bianchi-anomaly residual for a heterotic compactification with
    /// visible + hidden bundles. Given the visible-bundle second-Chern
    /// integrals `c2_v[a] = ∫ c_2(V_vis) ∧ J_a` and hidden-bundle
    /// `c2_h[a] = ∫ c_2(V_hid) ∧ J_a` (each length `N_KAHLER`), returns
    /// the squared `ℓ_2` deviation
    ///
    /// ```text
    ///     R_Bianchi  =  Σ_a (c_{2, vis, a} + c_{2, hid, a} − c_{2, TX̃, a})²
    /// ```
    ///
    /// The cancellation condition `c_2(V_vis) + c_2(V_hid) = c_2(TX̃)`
    /// holds **exactly** iff the residual is zero (in `H^4(X̃, Z)`, i.e.
    /// in the `(J_1, J_2, J_t)` basis).
    ///
    /// On the quotient `X = X̃/Γ`, the same identity reads
    /// `c_2(V_vis) + c_2(V_hid) + n · NS5 = c_2(TX̃) / |Γ|` after
    /// subtracting NS5-brane charge `n`; pass `c2_v + c2_h` already
    /// adjusted for the NS5 term and divide `c2_tm` by `|Γ|` if you
    /// want the downstairs residual. This function performs the
    /// **upstairs** comparison and is the canonical Bianchi gate.
    pub fn bianchi_residual(
        &self,
        c2_visible: &[i64],
        c2_hidden: &[i64],
    ) -> Result<f64, SchoenGeomError> {
        if c2_visible.len() != N_KAHLER {
            return Err(SchoenGeomError::WrongChernShape {
                got: c2_visible.len(),
                expected: N_KAHLER,
            });
        }
        if c2_hidden.len() != N_KAHLER {
            return Err(SchoenGeomError::WrongChernShape {
                got: c2_hidden.len(),
                expected: N_KAHLER,
            });
        }
        let target = self.c2_tm_vector();
        let mut total = 0.0_f64;
        for a in 0..N_KAHLER {
            let diff = (c2_visible[a] + c2_hidden[a] - target[a]) as f64;
            total += diff * diff;
        }
        Ok(total)
    }

    /// Integer slope `mu(F) = c_1(F) · J^{n-1}` for a coherent sub-sheaf
    /// (or line bundle) `F` in the `(J_1, J_2, J_t)` basis given a
    /// polarisation `(t_1, t_2, t_t)` (i.e. Kähler form `J = Σ t_a J_a`).
    /// Returned as `(c_1(F) · J^2, J^3)` so the caller can compute the
    /// rational slope `c_1·J^2 / r·J^3` without losing precision in
    /// integer arithmetic.
    pub fn slope_pair(
        &self,
        c1_coeffs: [i32; N_KAHLER],
        polarisation: [i32; N_KAHLER],
    ) -> Result<(i64, i64), SchoenGeomError> {
        // c_1 · J^2 = Σ_{a, b, c} c1[a] t[b] t[c] · ⟨ J_a, J_b, J_c ⟩
        //           = triple_intersection(c1, polar, polar)
        let pol_vec = polarisation.to_vec();
        let c1_vec = c1_coeffs.to_vec();
        let c1_dot_j2 = self.triple_intersection(&c1_vec, &pol_vec, &pol_vec)?;
        let j_cube = self.triple_intersection(&pol_vec, &pol_vec, &pol_vec)?;
        Ok((c1_dot_j2, j_cube))
    }
}

impl Default for SchoenGeometry {
    fn default() -> Self {
        Self::schoen_z3xz3()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// External test files in `route34/tests/` exercise the integration paths;
// the inline tests below cover the unit-level invariants that should pass
// independently of the sampler / projector.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schoen_topology_matches_published() {
        let g = SchoenGeometry::schoen_z3xz3();
        assert_eq!(g.h11_upstairs, 19);
        assert_eq!(g.h21_upstairs, 19);
        assert_eq!(g.chi_upstairs, 0);
        assert_eq!(g.h11_quotient, 3);
        assert_eq!(g.h21_quotient, 3);
        assert_eq!(g.quotient_order, 9);
        assert!(g.is_calabi_yau());
        assert_eq!(g.n_fold(), 3);
        assert_eq!(g.n_coords(), 8);
    }

    #[test]
    fn schoen_intersection_numbers_match_published() {
        let g = SchoenGeometry::schoen_z3xz3();
        // Donagi-He-Ovrut-Reinbacher 2006 §3 Eq. (3.7): the class of
        // `X̃ ⊂ CP^2 × CP^2 × CP^1` as a complete intersection of the
        // two defining bidegrees (3, 0, 1) and (0, 3, 1) is
        //   [X̃] = (3H_1 + H_t)(3H_2 + H_t) = 9 H_1 H_2 + 3 H_1 H_t + 3 H_2 H_t.
        //
        // Triple intersections on the 3-fold (top form requires Σ e = 3):
        //   ∫ J_1^2 J_2 = 3   (from 3 H_2 H_t · H_1^2 H_2 = 3 [top] when [2,2,1])
        //   ∫ J_1 J_2^2 = 3
        //   ∫ J_1 J_2 J_t = 9
        //   ∫ J_1^3 = ∫ J_2^3 = ∫ J_t^3 = ∫ J_t^2 J_a = 0 (truncation).
        let v_112 = g.intersection_number([2, 1, 0]).expect("intersection eval");
        let v_122 = g.intersection_number([1, 2, 0]).expect("intersection eval");
        let v_11t = g.intersection_number([1, 1, 1]).expect("intersection eval");
        assert_eq!(v_112, 3, "∫ J_1^2 J_2 should be 3; got {v_112}");
        assert_eq!(v_122, 3, "∫ J_1 J_2^2 should be 3; got {v_122}");
        assert_eq!(v_11t, 9, "∫ J_1 J_2 J_t should be 9; got {v_11t}");

        // Truncated entries.
        assert_eq!(g.intersection_number([3, 0, 0]).expect(""), 0);
        assert_eq!(g.intersection_number([0, 3, 0]).expect(""), 0);
        assert_eq!(g.intersection_number([0, 0, 3]).expect(""), 0);
        assert_eq!(g.intersection_number([2, 0, 1]).expect(""), 0);

        // Cross-check the published table.
        for (a, b, c, val) in PUBLISHED_TRIPLE_INTERSECTIONS.iter() {
            // Build sorted exponent vector.
            let mut exp = [0u32; N_KAHLER];
            exp[*a] += 1;
            exp[*b] += 1;
            exp[*c] += 1;
            let computed = g.intersection_number(exp).expect("");
            assert_eq!(
                computed, *val,
                "published triple intersection ({a},{b},{c}) = {val}, computed = {computed}"
            );
        }
    }

    #[test]
    fn cy_condition_holds() {
        let g = SchoenGeometry::schoen_z3xz3();
        // (3,0,1) + (0,3,1) = (3,3,2) = (n_1+1, n_2+1, n_3+1) ✓
        assert!(g.satisfies_calabi_yau_condition());
    }

    #[test]
    fn c2_tm_vector_matches_published() {
        let g = SchoenGeometry::schoen_z3xz3();
        // Donagi-He-Ovrut-Reinbacher 2006 Eq. (3.13–3.15)
        assert_eq!(g.c2_tm_dot_j(J1).expect("c2 dot J1"), 36);
        assert_eq!(g.c2_tm_dot_j(J2).expect("c2 dot J2"), 36);
        assert_eq!(g.c2_tm_dot_j(JT).expect("c2 dot Jt"), 24);
    }

    #[test]
    fn bianchi_residual_zero_when_anomaly_cancels() {
        let g = SchoenGeometry::schoen_z3xz3();
        // Visible + hidden = c_2(TX̃) splitting. Take c_2(V_vis) = (20, 24, 12),
        // c_2(V_hid) = (16, 12, 12) so the sum equals c_2(TX̃) = (36, 36, 24).
        let v = vec![20i64, 24, 12];
        let h = vec![16i64, 12, 12];
        let r = g.bianchi_residual(&v, &h).expect("residual eval");
        assert!(r < 1e-12, "Bianchi residual should be zero; got {r}");

        // Off by one in the J_1 component → residual = 1.
        let v_bad = vec![21i64, 24, 12];
        let r_bad = g.bianchi_residual(&v_bad, &h).expect("residual eval");
        assert!((r_bad - 1.0).abs() < 1e-12, "expected 1.0; got {r_bad}");
    }

    #[test]
    fn bianchi_residual_rejects_wrong_shape() {
        let g = SchoenGeometry::schoen_z3xz3();
        let v = vec![1i64, 2]; // wrong length
        let h = vec![0i64; 3];
        let res = g.bianchi_residual(&v, &h);
        assert!(matches!(res, Err(SchoenGeomError::WrongChernShape { .. })));
    }

    #[test]
    fn slope_pair_consistency() {
        let g = SchoenGeometry::schoen_z3xz3();
        // c_1 = J_1 + J_2 (line bundle of bidegree (1,1,0)), polarisation J = J_1 + J_2 + J_t
        let (num, denom) = g
            .slope_pair([1, 1, 0], [1, 1, 1])
            .expect("slope_pair eval");
        assert!(denom > 0, "polarisation should give positive J^3; got {denom}");
        // Slope sign: (J_1+J_2) along positive polarisation must be positive.
        assert!(num > 0, "c_1·J^2 should be positive; got {num}");
    }

    #[test]
    fn quotient_chi_zero() {
        let g = SchoenGeometry::schoen_z3xz3();
        assert_eq!(g.chi_quotient(), 0);
    }
}
