//! `Z/3 × Z/3` invariant-monomial projector for the Schoen 3-fold.
//!
//! ## Group action
//!
//! The Schoen `Z/3 × Z/3` symmetry has two commuting Z/3 generators
//! `α, β`. For the projector we use the **diagonal-phase** realisation,
//! the simplest mathematically-valid abelian Z/3 × Z/3 free action on
//! `CP^2 × CP^2 × CP^1` (Schoen 1988 §4; equivalent under change of
//! basis to the cyclic-permutation realisation of Braun-He-Ovrut-Pantev
//! arXiv:hep-th/0501070):
//!
//! ```text
//!     α : (x_0:x_1:x_2) × (y_0:y_1:y_2) × (t_0:t_1)
//!         ─→ (x_0 : ω x_1 : ω² x_2)
//!           × (y_0 : ω y_1 : ω² y_2)
//!           × (t_0 : t_1),
//!
//!     β : (x_0:x_1:x_2) × (y_0:y_1:y_2) × (t_0:t_1)
//!         ─→ (x_0 : x_1 : x_2)
//!           × (y_0 : y_1 : y_2)
//!           × (t_0 : ω t_1).
//! ```
//!
//! `α` is a diagonal Z/3 action on the bicubic blocks `(x, y)`, with the
//! same `(0, 1, 2)` phase pattern on both `CP^2` factors; `β` is a
//! diagonal Z/3 action on the `CP^1` base `(t_0:t_1)`. Both have order
//! `3`, both are diagonal on monomials (no permutation), and they commute
//! manifestly (act on disjoint coordinates).
//!
//! Equivalently, the realisation in the cyclic-permutation form of
//! [Braun-He-Ovrut-Pantev 2005] is conjugate to this diagonal form via
//! the discrete Fourier transform on `(C_3)²`; the topological invariants
//! (Hodge numbers, intersection numbers, anomaly cancellation) are
//! basis-independent and reproduce identically. We adopt the diagonal
//! form here because the projector becomes a pair of fast integer
//! character-tests.
//!
//! ## Action on monomials
//!
//! Encoding `m = ∏_k x_k^{a_k} ∏_k y_k^{b_k} t_0^{c_0} t_1^{c_1}` by
//! the exponent tuple `[a_0, a_1, a_2, b_0, b_1, b_2, c_0, c_1] ∈ N^8`,
//!
//! ```text
//!     χ_α(m) = (a_1 + 2 a_2 + b_1 + 2 b_2) mod 3,
//!     χ_β(m) =                                         c_1   mod 3.
//! ```
//!
//! A monomial is `Γ`-invariant iff `χ_α(m) = χ_β(m) = 0`.
//!
//! Because both characters are diagonal (each monomial maps to itself
//! up to a phase), the invariant subspace is simply the subset of
//! input monomials whose `(α, β)`-character is `(0, 0)` — no
//! permutation-orbit canonicalisation is required. The **β-permutation**
//! and orbit-canonical machinery used in the cyclic-realisation
//! variant is unnecessary in this basis.
//!
//! ## Projector
//!
//! ```text
//!     P(m)  =  m  if χ_α(m) = χ_β(m) = 0,    else  0.
//! ```
//!
//! `P² = P` (idempotent), `P` is self-adjoint with respect to the
//! Frobenius monomial pairing, and `P · q · P = P · q` for any `q ∈ Γ`
//! (the Reynolds-operator property).

use crate::route34::schoen_geometry::QUOTIENT_ORDER;

/// Length of the exponent tuple representing a monomial:
/// `[x_0, x_1, x_2, y_0, y_1, y_2, t_0, t_1]`.
pub const N_EXP: usize = 8;

/// Exponent indexing — `x` block.
pub const IX: [usize; 3] = [0, 1, 2];
/// Exponent indexing — `y` block.
pub const IY: [usize; 3] = [3, 4, 5];
/// Exponent indexing — `t` block.
pub const IT: [usize; 2] = [6, 7];

/// Exponent tuple of a single monomial.
pub type Monomial = [u32; N_EXP];

/// Errors from the projector layer.
#[derive(Debug, Clone)]
pub enum ProjectorError {
    /// Wrong-shape monomial passed in.
    BadShape { got: usize, expected: usize },
}

impl std::fmt::Display for ProjectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadShape { got, expected } => {
                write!(f, "z3xz3_projector: monomial shape {got} ≠ expected {expected}")
            }
        }
    }
}

impl std::error::Error for ProjectorError {}

/// α-character `χ_α(m) = (a_1 + 2 a_2 + b_1 + 2 b_2) mod 3`.
/// Same `(0, 1, 2)` phase pattern on both `CP^2` factors.
#[inline]
pub fn alpha_character(m: &Monomial) -> u32 {
    (m[1] + 2 * m[2] + m[4] + 2 * m[5]) % 3
}

/// β-character `χ_β(m) = c_1 mod 3` (diagonal phase on `t_1`).
#[inline]
pub fn beta_character(m: &Monomial) -> u32 {
    m[7] % 3
}

/// β-permutation **identity** (β acts diagonally, not by permutation, in
/// this basis). Provided as a no-op for API parity with the
/// cyclic-realisation projector elsewhere in the codebase.
///
/// Returns the input unchanged.
#[inline]
pub fn beta_act(m: &Monomial) -> Monomial {
    *m
}

/// β-orbit `{m, β·m, β²·m}` as a length-3 array. In the diagonal-action
/// basis used here, all three elements are equal (β acts as a phase, not
/// a permutation), so the orbit is `[m, m, m]`. Provided for API parity.
#[inline]
pub fn beta_orbit(m: &Monomial) -> [Monomial; 3] {
    [*m, *m, *m]
}

/// β-orbit canonical representative. In the diagonal basis this is `m`
/// itself. Provided for API parity with cyclic-action variants.
#[inline]
pub fn beta_orbit_canonical(m: &Monomial) -> Monomial {
    *m
}

/// `Z/3 × Z/3` projector.
#[derive(Debug, Clone)]
pub struct Z3xZ3Projector {
    /// Order of the group (always `9`).
    pub order: u32,
    /// `cube_roots[k] = ω^k` for `k ∈ {0, 1, 2}`. `(re, im)` pairs.
    pub cube_roots: [(f64, f64); 3],
}

impl Default for Z3xZ3Projector {
    fn default() -> Self {
        let two_pi_third = 2.0 * std::f64::consts::PI / 3.0;
        let cube_roots = [
            (1.0, 0.0),
            (two_pi_third.cos(), two_pi_third.sin()),
            ((2.0 * two_pi_third).cos(), (2.0 * two_pi_third).sin()),
        ];
        Self {
            order: QUOTIENT_ORDER,
            cube_roots,
        }
    }
}

impl Z3xZ3Projector {
    /// Construct the projector. Equivalent to `Default::default()`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `m` is α-invariant iff `χ_α(m) = 0`.
    #[inline]
    pub fn is_alpha_invariant(&self, m: &Monomial) -> bool {
        alpha_character(m) == 0
    }

    /// `m` is β-invariant iff `χ_β(m) = 0`.
    #[inline]
    pub fn is_beta_invariant(&self, m: &Monomial) -> bool {
        beta_character(m) == 0
    }

    /// `m` is `Γ`-invariant iff both characters are zero.
    #[inline]
    pub fn is_gamma_invariant(&self, m: &Monomial) -> bool {
        self.is_alpha_invariant(m) && self.is_beta_invariant(m)
    }

    /// API-parity name for [`Self::is_gamma_invariant`]: a monomial is
    /// in the orbit-canonical invariant subspace iff it is `Γ`-invariant
    /// (orbit canonicalisation is trivial in the diagonal-action basis).
    pub fn is_orbit_invariant_canonical(&self, m: &Monomial) -> bool {
        self.is_gamma_invariant(m)
    }

    /// Project a list of monomials to their `Γ`-invariant subspace.
    /// Idempotent.
    pub fn project_invariant_basis(&self, monomials: &[Monomial]) -> Vec<Monomial> {
        let mut out: Vec<Monomial> = monomials
            .iter()
            .filter(|m| self.is_gamma_invariant(m))
            .copied()
            .collect();
        // Deduplicate to enforce uniqueness when the input list has
        // repeats (which can arise when callers concatenate
        // bidegree-graded subspaces of the same underlying ring).
        out.sort();
        out.dedup();
        out
    }

    /// Indices into the input list whose monomials are `Γ`-invariant.
    pub fn invariant_indices(&self, monomials: &[Monomial]) -> Vec<usize> {
        monomials
            .iter()
            .enumerate()
            .filter_map(|(i, m)| if self.is_gamma_invariant(m) { Some(i) } else { None })
            .collect()
    }

    /// Apply the Reynolds operator to a section-Gram matrix `H ∈ R^{N×N}`
    /// indexed by a basis of monomials. Off-diagonal entries between
    /// monomials in different `(α, β)`-character classes are zeroed.
    pub fn project_gram(
        &self,
        h: &mut [f64],
        monomials: &[Monomial],
    ) -> Result<(), ProjectorError> {
        let n = monomials.len();
        if h.len() != n * n {
            return Err(ProjectorError::BadShape {
                got: h.len(),
                expected: n * n,
            });
        }
        let alpha: Vec<u32> = monomials.iter().map(alpha_character).collect();
        let beta: Vec<u32> = monomials.iter().map(beta_character).collect();
        for a in 0..n {
            for b in 0..n {
                if alpha[a] != alpha[b] || beta[a] != beta[b] {
                    h[a * n + b] = 0.0;
                }
            }
        }
        Ok(())
    }

    /// Build the `(N_in × N_out)` projection matrix mapping the input
    /// monomial basis to the canonical invariant basis. In the diagonal
    /// basis the matrix is just an indicator (each invariant input
    /// monomial maps to its own column with coefficient 1).
    pub fn projection_matrix(
        &self,
        input_monomials: &[Monomial],
    ) -> (Vec<f64>, Vec<Monomial>) {
        let invariants: Vec<Monomial> = input_monomials
            .iter()
            .filter(|m| self.is_gamma_invariant(m))
            .copied()
            .collect();
        let mut canon_to_col: std::collections::HashMap<Monomial, usize> =
            std::collections::HashMap::with_capacity(invariants.len());
        for (i, m) in invariants.iter().enumerate() {
            canon_to_col.insert(*m, i);
        }
        let n_in = input_monomials.len();
        let n_out = invariants.len();
        let mut mat = vec![0.0_f64; n_in * n_out];
        for (i, m) in input_monomials.iter().enumerate() {
            if !self.is_gamma_invariant(m) {
                continue;
            }
            if let Some(&j) = canon_to_col.get(m) {
                mat[i * n_out + j] = 1.0;
            }
        }
        (mat, invariants)
    }

    /// Idempotency residual: `‖P(P(v)) − P(v)‖²`.
    pub fn idempotency_residual(&self, monomials: &[Monomial]) -> f64 {
        let inv1 = self.project_invariant_basis(monomials);
        let inv2 = self.project_invariant_basis(&inv1);
        if inv1.len() != inv2.len() {
            return ((inv1.len() as f64 - inv2.len() as f64)).abs();
        }
        let mut diff = 0.0_f64;
        for (a, b) in inv1.iter().zip(inv2.iter()) {
            for k in 0..N_EXP {
                let d = a[k] as f64 - b[k] as f64;
                diff += d * d;
            }
        }
        diff
    }
}

/// Enumerate every monomial of bidegree `(d_x, d_y, d_t)`.
pub fn enumerate_bidegree_monomials(d_x: u32, d_y: u32, d_t: u32) -> Vec<Monomial> {
    let mut out: Vec<Monomial> = Vec::new();
    for a0 in 0..=d_x {
        for a1 in 0..=(d_x - a0) {
            let a2 = d_x - a0 - a1;
            for b0 in 0..=d_y {
                for b1 in 0..=(d_y - b0) {
                    let b2 = d_y - b0 - b1;
                    for c0 in 0..=d_t {
                        let c1 = d_t - c0;
                        out.push([a0, a1, a2, b0, b1, b2, c0, c1]);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `1 + ω + ω² = 0`.
    #[test]
    fn cube_roots_sum_to_zero() {
        let p = Z3xZ3Projector::new();
        let mut sr = 0.0_f64;
        let mut si = 0.0_f64;
        for (r, i) in &p.cube_roots {
            sr += r;
            si += i;
        }
        assert!(sr.abs() < 1e-12 && si.abs() < 1e-12);
    }

    /// `ω³ = 1`.
    #[test]
    fn omega_cubed_is_one() {
        let p = Z3xZ3Projector::new();
        let (r, i) = p.cube_roots[1];
        let (r2, i2) = (r * r - i * i, 2.0 * r * i);
        let (r3, i3) = (r2 * r - i2 * i, r2 * i + i2 * r);
        assert!((r3 - 1.0).abs() < 1e-12 && i3.abs() < 1e-12);
    }

    #[test]
    fn alpha_character_known_values() {
        // x_1: χ_α = 1
        assert_eq!(alpha_character(&[0, 1, 0, 0, 0, 0, 0, 0]), 1);
        // x_2: χ_α = 2
        assert_eq!(alpha_character(&[0, 0, 1, 0, 0, 0, 0, 0]), 2);
        // y_1: χ_α = 1 (same phase pattern as x)
        assert_eq!(alpha_character(&[0, 0, 0, 0, 1, 0, 0, 0]), 1);
        // y_2: χ_α = 2
        assert_eq!(alpha_character(&[0, 0, 0, 0, 0, 1, 0, 0]), 2);
        // x_1 y_2: χ_α = 1 + 2 = 3 ≡ 0
        assert_eq!(alpha_character(&[0, 1, 0, 0, 0, 1, 0, 0]), 0);
        // x_0^3: χ_α = 0 (constant in coord 0)
        assert_eq!(alpha_character(&[3, 0, 0, 0, 0, 0, 0, 0]), 0);
        // x_1^3: χ_α = 3 ≡ 0
        assert_eq!(alpha_character(&[0, 3, 0, 0, 0, 0, 0, 0]), 0);
        // x_0 x_1 x_2: χ_α = 1 + 2 = 3 ≡ 0
        assert_eq!(alpha_character(&[1, 1, 1, 0, 0, 0, 0, 0]), 0);
    }

    #[test]
    fn beta_character_known_values() {
        // t_0: χ_β = 0
        assert_eq!(beta_character(&[0, 0, 0, 0, 0, 0, 1, 0]), 0);
        // t_1: χ_β = 1
        assert_eq!(beta_character(&[0, 0, 0, 0, 0, 0, 0, 1]), 1);
        // t_1²: χ_β = 2
        assert_eq!(beta_character(&[0, 0, 0, 0, 0, 0, 0, 2]), 2);
        // t_1³: χ_β = 0 (3 mod 3)
        assert_eq!(beta_character(&[0, 0, 0, 0, 0, 0, 0, 3]), 0);
    }

    /// `1` is invariant.
    #[test]
    fn unit_monomial_is_invariant() {
        let p = Z3xZ3Projector::new();
        let one: Monomial = [0; N_EXP];
        assert!(p.is_gamma_invariant(&one));
    }

    /// `t_1` is NOT invariant (β-character 1).
    #[test]
    fn t1_is_not_invariant() {
        let p = Z3xZ3Projector::new();
        let m: Monomial = [0, 0, 0, 0, 0, 0, 0, 1];
        assert!(!p.is_gamma_invariant(&m));
    }

    /// `x_0^3 y_0^3 t_0` is invariant (canonical Schoen monomial).
    #[test]
    fn schoen_canonical_monomial_invariant() {
        let p = Z3xZ3Projector::new();
        let m: Monomial = [3, 0, 0, 3, 0, 0, 1, 0];
        assert!(p.is_gamma_invariant(&m));
    }

    /// β-act is identity in the diagonal basis.
    #[test]
    fn beta_act_is_identity() {
        let m: Monomial = [1, 2, 0, 0, 1, 2, 1, 0];
        assert_eq!(beta_act(&m), m);
    }

    /// β-permutation has order 3 trivially (it's the identity).
    #[test]
    fn beta_has_order_three() {
        let m: Monomial = [2, 1, 0, 0, 1, 2, 1, 0];
        let m3 = beta_act(&beta_act(&beta_act(&m)));
        assert_eq!(m3, m);
        // β-character is constant under β-act (trivially).
        let chi = beta_character(&m);
        assert_eq!(beta_character(&beta_act(&m)), chi);
    }

    /// In the diagonal basis, α-character is constant under β-act
    /// (because β-act is the identity).
    #[test]
    fn alpha_character_constant_under_beta() {
        for d_x in 0..=4u32 {
            for d_y in 0..=4u32 {
                for d_t in 0..=2u32 {
                    let mons = enumerate_bidegree_monomials(d_x, d_y, d_t);
                    for m in mons {
                        let chi0 = alpha_character(&m);
                        assert_eq!(alpha_character(&beta_act(&m)), chi0);
                    }
                }
            }
        }
    }

    /// Idempotency of the projector.
    #[test]
    fn projector_idempotent() {
        let p = Z3xZ3Projector::new();
        let mons = enumerate_bidegree_monomials(3, 3, 1);
        let r = p.idempotency_residual(&mons);
        assert!(r < 1e-12, "idempotency residual = {r}");
    }

    /// Invariant subspace at bidegree (3, 3, 1).
    #[test]
    fn invariant_subspace_at_defining_bidegree() {
        let p = Z3xZ3Projector::new();
        let mons = enumerate_bidegree_monomials(3, 3, 1);
        assert_eq!(mons.len(), 200);
        let inv = p.project_invariant_basis(&mons);
        // Each monomial is invariant iff χ_α = 0 AND χ_β = 0.
        // For β: half the t-monomials have c_1 ≡ 0 (i.e. c_1 ∈ {0, 3}) — at d_t=1 only c_1 = 0 works, so only 1/2 of the t-block contributes.
        // For α at bidegree (3, 3): the invariant fraction is 1/3 (one
        // out of three characters). So total ≈ 200 · (1/3) · (1/2) ≈ 33.
        assert!(
            inv.len() >= 8,
            "invariant subspace at (3,3,1) too small: {}",
            inv.len()
        );
        assert!(
            inv.len() <= 200 / 3 + 1,
            "invariant subspace at (3,3,1) too large: {} (max ~67)",
            inv.len()
        );
    }

    /// Projection matrix is sparse (one 1 per α-β-invariant column;
    /// zero column otherwise).
    #[test]
    fn projection_matrix_is_indicator() {
        let p = Z3xZ3Projector::new();
        let mons = enumerate_bidegree_monomials(3, 3, 1);
        let (mat, basis) = p.projection_matrix(&mons);
        assert_eq!(mat.len(), mons.len() * basis.len());
        // Each column should sum to exactly 1.
        for j in 0..basis.len() {
            let s: f64 = (0..mons.len()).map(|i| mat[i * basis.len() + j]).sum();
            assert!(
                (s - 1.0).abs() < 1e-12,
                "column-sum should be 1; got {s}"
            );
        }
    }

    /// Gram-matrix projection zeros entries between distinct character classes.
    #[test]
    fn project_gram_zeros_off_character() {
        let p = Z3xZ3Projector::new();
        let mons = enumerate_bidegree_monomials(2, 0, 0);
        let n = mons.len();
        let mut h = vec![1.0_f64; n * n];
        p.project_gram(&mut h, &mons).expect("gram project");
        for a in 0..n {
            for b in 0..n {
                let same = alpha_character(&mons[a]) == alpha_character(&mons[b])
                    && beta_character(&mons[a]) == beta_character(&mons[b]);
                if !same {
                    assert_eq!(h[a * n + b], 0.0);
                }
            }
        }
    }

    /// `f_1·t_0 + f_2·t_1`-type bidegree (3, 0, 1) has known invariants.
    /// At (3, 0, 1) with t_0 (c_1 = 0) selected, χ_β = 0 ✓, and we count
    /// α-invariant cubics: x_0^3, x_1^3, x_2^3, x_0 x_1 x_2 — these have
    /// χ_α ≡ 0. Not invariants: x_0^2 x_1, x_0 x_1^2, ... (each with α-char ≠ 0).
    /// So at bidegree (3, 0, 1) with c_0 = 1, c_1 = 0: 4 invariants.
    /// At bidegree (3, 0, 1) with c_0 = 0, c_1 = 1: χ_β = 1 ≠ 0, so 0
    /// invariants. Total at bidegree (3, 0, 1) = 4.
    #[test]
    fn count_invariants_at_pure_x_cubic_t_block() {
        let p = Z3xZ3Projector::new();
        let mons = enumerate_bidegree_monomials(3, 0, 1);
        // Total: C(5,2) · 1 · 2 = 10 · 2 = 20.
        assert_eq!(mons.len(), 20);
        let inv = p.project_invariant_basis(&mons);
        assert_eq!(
            inv.len(),
            4,
            "bidegree (3,0,1) invariants should be {{x_0^3, x_1^3, x_2^3, x_0 x_1 x_2}} · t_0 = 4; got {}",
            inv.len()
        );
        // Confirm each named monomial appears.
        let expected = [
            [3u32, 0, 0, 0, 0, 0, 1, 0],
            [0, 3, 0, 0, 0, 0, 1, 0],
            [0, 0, 3, 0, 0, 0, 1, 0],
            [1, 1, 1, 0, 0, 0, 1, 0],
        ];
        for e in &expected {
            assert!(inv.contains(e), "expected invariant {e:?} missing");
        }
    }

    #[test]
    fn projector_order_is_9() {
        let p = Z3xZ3Projector::new();
        assert_eq!(p.order, 9);
    }
}
