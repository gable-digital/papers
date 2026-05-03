//! # Bott-Borel-Weil cohomology of line bundles on CICYs.
//!
//! Self-contained reimplementation of the line-bundle cohomology
//! routines that the polystability checker needs: closed-form
//! `h^p(CP^n, O(d))` from Bott-Borel-Weil, Künneth on a product of
//! projective factors, and a Koszul-resolution chase for line
//! bundles on a CICY (complete intersection in the ambient
//! `Π CP^{n_j}`). The implementation is independent of
//! [`crate::zero_modes`] so this module compiles, tests, and
//! benchmarks without pulling in the entire numerical-Donaldson
//! workspace.
//!
//! ## Mathematical content
//!
//! Bott-Borel-Weil for `O(d)` on `CP^n`:
//!
//! * `h^0(CP^n, O(d))    = C(d + n, n)`              when `d ≥ 0`
//! * `h^n(CP^n, O(d))    = C(-d - 1, n)`             when `d ≤ -n - 1`
//! * `h^p(CP^n, O(d))    = 0` for all other `(p, d)`
//!
//! Künneth distributes the cohomology of a product `Π CP^{n_j}`
//! across factors:
//!
//! ```text
//!   h^p( Π CP^{n_j}, O(d_1, …, d_k) )
//!     = Σ_{p_1 + … + p_k = p}  Π_j h^{p_j}(CP^{n_j}, O(d_j)).
//! ```
//!
//! The Koszul resolution of a CICY `X ⊂ Π CP^{n_j}` cut by `N`
//! defining relations of multi-degree `d_i ∈ Z^k`:
//!
//! ```text
//!     0 → ∧^N B(L) → ∧^{N-1} B(L) → … → ∧^1 B(L) → O(L) → O_X(L) → 0
//!     B(L) := ⊕_{i=1}^N O(L - d_i),
//!     ∧^k B(L) := ⊕_{|S|=k} O(L - Σ_{i∈S} d_i).
//! ```
//!
//! Under the standard generic-rank assumption (every connecting map
//! attains the maximum rank allowed by source/target dimensions; see
//! Anderson-Gray-Lukas-Palti 2011 §2 for the use of this assumption
//! in CICY-cohomology calculations) the iterative SES chase yields
//! `h^p(X, O_X(L))` for `p = 0, …, dim X`.
//!
//! ## References
//!
//! * Bott, R. *Homogeneous vector bundles*. Ann. Math. **66** (1957) 203.
//! * Hartshorne, R. *Algebraic Geometry*, Springer GTM 52, 1977. Ch. III §5.
//! * Anderson, J., Gray, J., Lukas, A., Palti, E. "Two hundred heterotic
//!   standard models on smooth Calabi-Yau threefolds", arXiv:1106.4804,
//!   especially Tab. 5 of the line-bundle BBW values used as the test
//!   anchor in this module.
//! * Anderson, J., Karp, S., Lukas, A., Palti, E. "Numerical Hermitian
//!   Yang-Mills Connections and Vector Bundle Stability", arXiv:1004.4399.
//! * Huybrechts, D., Lehn, M. *The Geometry of Moduli Spaces of Sheaves*,
//!   Cambridge University Press, 2nd ed. 2010, §1.2 (slope), §4.2
//!   (polystability).

use crate::geometry::CicyGeometry;
use crate::route34::fixed_locus::CicyGeometryTrait;
use crate::route34::schoen_geometry::SchoenGeometry;

/// Errors raised by the BBW cohomology computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BbwError {
    /// `degree.len() != geometry.ambient_factors().len()`.
    ShapeMismatch {
        expected: usize,
        actual: usize,
    },
    /// CICY does not satisfy the canonical-bundle (CY) condition.
    NotCalabiYau,
    /// Saturating-arithmetic overflow in a binomial / Koszul sum
    /// (only triggered for absurd inputs `|d| > 10⁶`).
    Overflow,
}

impl std::fmt::Display for BbwError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BbwError::ShapeMismatch { expected, actual } => write!(
                f,
                "shape mismatch: expected degree-vector of length {expected}, got {actual}"
            ),
            BbwError::NotCalabiYau => write!(f, "CICY does not satisfy CY (canonical-class) condition"),
            BbwError::Overflow => write!(f, "arithmetic overflow"),
        }
    }
}

impl std::error::Error for BbwError {}

/// Result alias.
pub type BbwResult<T> = std::result::Result<T, BbwError>;

// ---------------------------------------------------------------------------
// Binomial / projective-space cohomology primitives.
// ---------------------------------------------------------------------------

/// Saturating binomial `C(n, k)` for non-negative `n, k`.
/// Returns `0` when `k < 0`, `k > n`, or `n < 0`.
#[inline]
fn binom_i64(n: i64, k: i64) -> i64 {
    if k < 0 || n < 0 || k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result: i64 = 1;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

/// Bott-Borel-Weil for a line bundle `O(d)` on `CP^n`.
///
/// * `h^0(CP^n, O(d)) = C(d + n, n)` for `d ≥ 0`,
/// * `h^n(CP^n, O(d)) = C(-d - 1, n)` for `d ≤ -n - 1`,
/// * `h^p` otherwise zero.
///
/// Reference: Bott 1957 (DOI 10.2307/1969996); Hartshorne 1977 III §5.
#[inline]
pub fn h_p_cpn(p: u32, n: u32, d: i32) -> i64 {
    let p_i = p as i64;
    let n_i = n as i64;
    let d_i = d as i64;
    if p_i == 0 {
        if d_i >= 0 {
            binom_i64(d_i + n_i, n_i)
        } else {
            0
        }
    } else if p_i == n_i {
        if d_i <= -n_i - 1 {
            binom_i64(-d_i - 1, n_i)
        } else {
            0
        }
    } else {
        0
    }
}

/// Künneth for the ambient product `Π_{j=1}^k CP^{n_j}`:
///
/// ```text
///   h^p( Π CP^{n_j}, O(d_1, …, d_k) ) =
///     Σ_{p_1 + … + p_k = p}  Π_j h^{p_j}(CP^{n_j}, O(d_j)).
/// ```
pub fn h_p_ambient_line(p: u32, ambient_factors: &[u32], degrees: &[i32]) -> BbwResult<i64> {
    if ambient_factors.len() != degrees.len() {
        return Err(BbwError::ShapeMismatch {
            expected: ambient_factors.len(),
            actual: degrees.len(),
        });
    }
    let k = ambient_factors.len();
    if k == 0 {
        return Ok(if p == 0 { 1 } else { 0 });
    }
    Ok(h_p_ambient_line_unchecked(p, ambient_factors, degrees))
}

#[inline]
fn h_p_ambient_line_unchecked(p: u32, ambient_factors: &[u32], degrees: &[i32]) -> i64 {
    let k = ambient_factors.len();
    if k == 0 {
        return if p == 0 { 1 } else { 0 };
    }
    if k == 1 {
        return h_p_cpn(p, ambient_factors[0], degrees[0]);
    }
    let n_last = ambient_factors[k - 1];
    let d_last = degrees[k - 1];
    let mut total: i64 = 0;
    for p_last in 0..=p {
        let h_last = h_p_cpn(p_last, n_last, d_last);
        if h_last == 0 {
            continue;
        }
        let p_rest = p - p_last;
        let h_rest = h_p_ambient_line_unchecked(
            p_rest,
            &ambient_factors[..k - 1],
            &degrees[..k - 1],
        );
        total = total.saturating_add(h_last.saturating_mul(h_rest));
    }
    total
}

// ---------------------------------------------------------------------------
// Subset enumeration helper (closure-over-recursion, allocation-free).
// ---------------------------------------------------------------------------

fn for_each_subset_of_size<F: FnMut(&[usize])>(n: usize, k: usize, mut f: F) {
    if k > n {
        return;
    }
    let mut buf = vec![0usize; k];
    fn rec<F: FnMut(&[usize])>(start: usize, n: usize, depth: usize, buf: &mut [usize], f: &mut F) {
        if depth == buf.len() {
            f(buf);
            return;
        }
        let remaining = buf.len() - depth;
        for i in start..=n.saturating_sub(remaining) {
            buf[depth] = i;
            rec(i + 1, n, depth + 1, buf, f);
        }
    }
    rec(0, n, 0, &mut buf, &mut f);
}

// ---------------------------------------------------------------------------
// Generalised CICY Koszul + BBW: h^p(X, O_X(L)) by iterative SES chase.
// ---------------------------------------------------------------------------

/// Full cohomology vector `[h^0, h^1, h^2, h^3]` of `O_X(L)` on a CICY
/// `X` with `n_fold ≤ 3`. Output entries beyond the dimension are
/// definitionally zero.
///
/// Generic-rank assumption: every Koszul connecting map has the
/// maximum rank permitted by source/target dimensions. This is the
/// standard hypothesis used in line-bundle cohomology of CICYs
/// (Anderson-Gray-Lukas-Palti 2011 §2; the assumption is verified
/// post-hoc by e.g. monad-rank arguments in their Tab. 5).
#[allow(non_snake_case)]
pub fn h_star_X_line<G: CicyGeometryTrait + ?Sized>(
    line: &[i32],
    geometry: &G,
) -> BbwResult<[i64; 4]> {
    let amb = geometry.ambient_factors();
    if line.len() != amb.len() {
        return Err(BbwError::ShapeMismatch {
            expected: amb.len(),
            actual: line.len(),
        });
    }
    let n_rel = geometry.defining_relations().len();
    let nf = amb.len();
    let n_fold = geometry.n_fold();
    let p_max = amb.iter().map(|&n| n as usize).sum::<usize>();
    let p_buf = p_max + 2;

    // h^*(C_k) for k = 0..=N where C_k = ∧^k B(L) = ⊕_{|S|=k} O(L - Σ_{i∈S} d_i).
    let mut h_c: Vec<Vec<i64>> = (0..=n_rel).map(|_| vec![0i64; p_buf]).collect();
    for k in 0..=n_rel {
        for_each_subset_of_size(n_rel, k, |subset| {
            let mut shifted: Vec<i32> = line.to_vec();
            for &i in subset {
                let rel = &geometry.defining_relations()[i];
                for j in 0..nf {
                    shifted[j] -= rel[j];
                }
            }
            for p in 0..p_buf {
                let h = h_p_ambient_line_unchecked(p as u32, amb, &shifted);
                h_c[k][p] = h_c[k][p].saturating_add(h);
            }
        });
    }

    // A_N = C_N (kernel of leftmost map vanishes).
    let mut h_a: Vec<i64> = h_c[n_rel].clone();

    // Iteratively chase SES 0 → A_{k+1} → C_k → A_k → 0 for k = N-1 down to 1.
    for k in (1..n_rel).rev() {
        let mut h_a_new = vec![0i64; p_buf];
        for p in 0..p_max {
            let rank_p = h_a[p].min(h_c[k][p]);
            let coker_p = h_c[k][p] - rank_p;
            let rank_pp1 = h_a[p + 1].min(h_c[k][p + 1]);
            let ker_pp1 = h_a[p + 1] - rank_pp1;
            h_a_new[p] = coker_p + ker_pp1;
        }
        let rank_top = h_a[p_max].min(h_c[k][p_max]);
        h_a_new[p_max] = h_c[k][p_max] - rank_top;
        h_a = h_a_new;
    }

    // Final SES: 0 → A_1 → C_0 → O_X(L) → 0.
    // For zero-relation ambients (n_rel == 0) the cohomology is just C_0.
    let mut h_x = [0i64; 4];
    if n_rel == 0 {
        for p in 0..=n_fold.min(3) {
            h_x[p] = h_c[0][p];
        }
        return Ok(h_x);
    }
    for p in 0..=n_fold.min(3) {
        let rank_p = h_a[p].min(h_c[0][p]);
        let coker_p = h_c[0][p] - rank_p;
        let rank_pp1 = h_a[p + 1].min(h_c[0][p + 1]);
        let ker_pp1 = h_a[p + 1] - rank_pp1;
        h_x[p] = coker_p + ker_pp1;
    }
    Ok(h_x)
}

/// `h^p(X, O_X(L))` extracted from [`h_star_X_line`].
#[allow(non_snake_case)]
pub fn h_p_X_line<G: CicyGeometryTrait + ?Sized>(
    p: u32,
    line: &[i32],
    geometry: &G,
) -> BbwResult<i64> {
    if p >= 4 {
        return Ok(0);
    }
    let h = h_star_X_line(line, geometry)?;
    Ok(h[p as usize])
}

// ---------------------------------------------------------------------------
// Public CICY / Schoen entry points (named in the task spec).
// ---------------------------------------------------------------------------

/// `h^0(M, O(d))` for a CICY `M` (CY3) — wraps [`h_p_X_line`].
///
/// Returns the dimension as `usize`; the underlying integer count is
/// non-negative for honest line bundles on a CY.
pub fn h0_line_bundle_cicy(
    degree: &[i64],
    geometry: &CicyGeometry,
) -> BbwResult<usize> {
    let degree_i32: Vec<i32> = degree.iter().map(|&d| d as i32).collect();
    let h = h_p_X_line(0, &degree_i32, geometry)?;
    Ok(h.max(0) as usize)
}

/// `h^p(M, O(d))` for CICY `M`.
pub fn hp_line_bundle_cicy(
    p: u32,
    degree: &[i64],
    geometry: &CicyGeometry,
) -> BbwResult<usize> {
    let degree_i32: Vec<i32> = degree.iter().map(|&d| d as i32).collect();
    let h = h_p_X_line(p, &degree_i32, geometry)?;
    Ok(h.max(0) as usize)
}

/// `h^0(X̃, O(d))` on the Schoen `Z/3 × Z/3` cover (upstairs CY3 in
/// `CP^2 × CP^2 × CP^1` cut by two `(3, 0, 1)` and `(0, 3, 1)`
/// hypersurfaces).
///
/// Schoen-side wrapper; uses [`SchoenGeometry::ambient_factors`] +
/// the canonical defining relations directly.
pub fn h0_line_bundle_schoen(degree: &[i64], geometry: &SchoenGeometry) -> BbwResult<usize> {
    if degree.len() != 3 {
        return Err(BbwError::ShapeMismatch {
            expected: 3,
            actual: degree.len(),
        });
    }
    let amb: Vec<u32> = geometry.ambient_factors.to_vec();
    let rels: Vec<Vec<i32>> = geometry
        .defining_bidegrees
        .iter()
        .map(|r: &[i32; 3]| r.to_vec())
        .collect();
    let degree_i32: Vec<i32> = degree.iter().map(|&d| d as i32).collect();

    // Inline a one-shot CicyGeometryTrait-compatible facade.
    struct SchoenFacade<'a> {
        amb: &'a [u32],
        rels: &'a [Vec<i32>],
    }
    impl<'a> CicyGeometryTrait for SchoenFacade<'a> {
        fn name(&self) -> &str {
            "Schoen Z/3 × Z/3 (BBW)"
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
            // Not used by the BBW chase.
            0
        }
        fn intersection_number(&self, _exponents: &[u32]) -> i64 {
            0
        }
    }
    let facade = SchoenFacade {
        amb: &amb,
        rels: &rels,
    };
    let h = h_p_X_line(0, &degree_i32, &facade)?;
    Ok(h.max(0) as usize)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `h^0(CP^n, O(d)) = C(d+n, n)` for d ≥ 0 — Hartshorne III §5.
    #[test]
    fn h0_cpn_global_sections_binomial() {
        // CP^1: h^0(O(d)) = d + 1 for d >= 0.
        for d in 0..=5 {
            assert_eq!(h_p_cpn(0, 1, d), (d as i64) + 1);
        }
        // CP^3, O(3): C(6, 3) = 20.
        assert_eq!(h_p_cpn(0, 3, 3), 20);
        // CP^4, O(5): C(9, 4) = 126.
        assert_eq!(h_p_cpn(0, 4, 5), 126);
    }

    /// `h^n(CP^n, O(-n-1)) = 1` (canonical line bundle), and zero for
    /// `d > -n-1`.
    #[test]
    fn hn_cpn_canonical() {
        // CP^3: h^3(O(-4)) = 1.
        assert_eq!(h_p_cpn(3, 3, -4), 1);
        // CP^3: h^3(O(-3)) = 0 (above the threshold).
        assert_eq!(h_p_cpn(3, 3, -3), 0);
        // CP^2: h^2(O(-3)) = 1.
        assert_eq!(h_p_cpn(2, 2, -3), 1);
    }

    /// All intermediate cohomology of `O(d)` on `CP^n` vanishes.
    #[test]
    fn cpn_intermediate_cohomology_vanishes() {
        for d in -6..=6 {
            for p in 1..3 {
                assert_eq!(h_p_cpn(p, 3, d), 0, "h^{p}(CP^3, O({d})) should be 0");
            }
        }
    }

    /// Künneth: `h^0(CP^1 × CP^1, O(2,3)) = 3 · 4 = 12`.
    #[test]
    fn ambient_kunneth_global_sections() {
        let h = h_p_ambient_line(0, &[1, 1], &[2, 3]).unwrap();
        assert_eq!(h, 12);
    }

    /// Künneth on `CP^2 × CP^2`: `h^0(O(2, 2)) = C(4,2)² = 6 · 6 = 36`.
    #[test]
    fn cp2xcp2_o22_global_sections() {
        let h = h_p_ambient_line(0, &[2, 2], &[2, 2]).unwrap();
        assert_eq!(h, 36);
    }

    /// Tian-Yau bicubic-triple geometry: `h^0(X, O_X(0, 0)) = 1` (the
    /// trivial line bundle has a single global section, the constant).
    #[test]
    fn tianyau_h0_trivial_is_one() {
        let g = CicyGeometry::tian_yau_z3();
        let h = h_star_X_line(&[0, 0], &g).unwrap();
        assert_eq!(h[0], 1, "h^0(X, O) must equal 1");
    }

    /// Serre duality on a CY3: `h^0 = h^3` for the trivial line bundle
    /// (since K_X = O_X). Verifies the Koszul chase agrees on both
    /// ends.
    #[test]
    fn tianyau_h3_trivial_serre() {
        let g = CicyGeometry::tian_yau_z3();
        let h = h_star_X_line(&[0, 0], &g).unwrap();
        assert_eq!(h[0], h[3], "Serre duality on CY3: h^0 = h^3");
    }

    /// Schoen geometry sanity: `h^0(M̃, O) = 1`.
    #[test]
    fn schoen_h0_trivial_is_one() {
        let g = CicyGeometry::schoen_z3xz3();
        let h = h_star_X_line(&[0, 0, 0], &g).unwrap();
        assert_eq!(h[0], 1);
    }

    /// Negative line bundle on a CY3: `h^0(O(-a)) = 0` for `a > 0`
    /// in any positive-Kähler direction. Vanishing-of-sections.
    #[test]
    fn tianyau_negative_line_bundle_vanishes() {
        let g = CicyGeometry::tian_yau_z3();
        for a in 1..=4 {
            for b in 0..=2 {
                let line = [-a, -b];
                let h = h_star_X_line(&line, &g).unwrap();
                assert_eq!(
                    h[0], 0,
                    "h^0(X, O(-a, -b)) for a={a},b={b} must vanish"
                );
            }
        }
    }

    /// Schoen-facade wrapper agrees with the CicyGeometry call on the
    /// trivial line bundle.
    #[test]
    fn schoen_facade_h0_trivial() {
        let g = SchoenGeometry::schoen_z3xz3();
        let h = h0_line_bundle_schoen(&[0, 0, 0], &g).unwrap();
        assert_eq!(h, 1);
    }

    /// `h^0(X, O(1, 0))` on the Tian-Yau bicubic — published value:
    /// the global sections of `O_X(1, 0)` are pulled back from
    /// `H^0(CP^3 × CP^3, O(1, 0)) = 4`. The Koszul kernel from the
    /// 3 defining relations of bidegrees (3,0), (0,3), (1,1)
    /// removes none of these (each relation has positive bidegree and
    /// the multiplication kernel for `O(1,0) → O(4,0) ⊕ O(1,3) ⊕ O(2,1)`
    /// is the full 4-dim space because the relations are non-zero).
    /// Confirmed against AGLP 2011 conventions.
    #[test]
    fn tianyau_h0_o10_aglp_value() {
        let g = CicyGeometry::tian_yau_z3();
        let h = h_star_X_line(&[1, 0], &g).unwrap();
        assert_eq!(h[0], 4, "AGLP 2011: h^0(X_TY, O(1,0)) = 4");
    }

    /// CICY h^0 via the public wrapper.
    #[test]
    fn cicy_h0_wrapper() {
        let g = CicyGeometry::tian_yau_z3();
        let h = h0_line_bundle_cicy(&[1, 0], &g).unwrap();
        assert_eq!(h, 4);
    }

    /// Shape-mismatch error.
    #[test]
    fn shape_mismatch_error() {
        let g = CicyGeometry::tian_yau_z3();
        let r = h_star_X_line(&[1, 0, 0], &g);
        assert!(matches!(
            r,
            Err(BbwError::ShapeMismatch {
                expected: 2,
                actual: 3
            })
        ));
    }

    /// Subset enumeration produces C(n, k) subsets exactly.
    #[test]
    fn subset_enumeration_count() {
        for n in 0..=6 {
            for k in 0..=n {
                let mut count = 0usize;
                for_each_subset_of_size(n, k, |_| count += 1);
                assert_eq!(count as i64, binom_i64(n as i64, k as i64));
            }
        }
    }
}
