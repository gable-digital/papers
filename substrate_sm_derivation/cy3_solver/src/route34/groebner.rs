//! Buchberger's algorithm for Gröbner bases over the rationals (Q),
//! restricted to the dense, polynomial-tower subset needed by the route34
//! Calabi-Yau metric solvers (TY and Schoen).
//!
//! ## What this module provides
//!
//! * [`MonomialOrder`] — degree-lex / degree-revlex / lex monomial orders
//!   on `N`-variable monomials (`N = 8` for both TY and Schoen).
//! * [`Polynomial`] — sorted list of `(coeff, exponent_vector)` terms over
//!   `Q` (represented as `f64` rationals; cancellations are exact for the
//!   small-integer ideals appearing in the chapter-21 CICY problems, and
//!   are guarded by an explicit zero-tolerance after subtraction).
//! * [`s_polynomial`] — Buchberger's S-polynomial.
//! * [`divide_with_remainder`] — multivariate polynomial division
//!   `f = Σ q_i g_i + r` with `LM(r)` not divisible by any `LM(g_i)`.
//! * [`buchberger`] — Buchberger's algorithm with the standard
//!   product-criterion + LCM-pair filtering, multi-threaded over
//!   independent S-pair reductions via rayon.
//! * [`reduced_groebner`] — minimal reduced Gröbner basis (unique modulo
//!   the chosen monomial order, up to scalar normalisation of leading
//!   coefficients).
//!
//! ## Why `f64` for coefficients
//!
//! The two CY3 ideals we apply this to,
//!
//!   TY      `(z_0³+z_1³+z_2³+z_3³, w_0³+w_1³+w_2³+w_3³, Σ z_i w_i)`,
//!   Schoen  `(F_1, F_2)` of bidegree `(3, 0, 1)` and `(0, 3, 1)`,
//!
//! both have `±1` integer coefficients, and Buchberger reduction over
//! these generators can only produce coefficients with denominators
//! formed from products of leading coefficients (all `±1`). All
//! intermediate coefficients therefore land on small integers / their
//! reciprocals, which `f64` represents exactly. We assert at every
//! reduction step that the remainder coefficients remain bounded; if
//! they ever exceed a sanity bound the algorithm reports
//! [`GroebnerError::CoefficientBlowup`] rather than silently rounding.
//!
//! ## References
//!
//! * Cox, Little, O'Shea, *Ideals, Varieties, and Algorithms*, 4th ed.
//!   (Springer 2015), Ch. 2 §§5-9. ISBN 978-3-319-16720-6.
//! * Buchberger, "An algorithmic criterion for the solvability of a
//!   system of algebraic equations", *Aequationes Math.* 4 (1970)
//!   374-383, DOI 10.1007/BF01844169.
//! * Kapur, "An approach for solving systems of parametric polynomial
//!   equations", in *Principles and Practice of Constraint Programming*,
//!   MIT Press 1986, ISBN 0262024144.

use std::cmp::Ordering;

use rayon::prelude::*;

/// Number of variables. The route34 ideals (TY, Schoen) both live in
/// 8 variables (`z_0..z_3, w_0..w_3` for TY; `x_0..x_2, y_0..y_2, t_0,
/// t_1` for Schoen).
pub const NVAR: usize = 8;

/// Exponent vector of a monomial.
pub type Exponent = [u32; NVAR];

/// Tolerance below which a coefficient is treated as exactly zero.
/// All coefficients in the route34 ideals are integers or simple
/// rationals, so any non-zero coefficient must be far above this.
pub const ZERO_TOL: f64 = 1.0e-12;

/// Sanity bound on individual coefficients. The TY/Schoen reductions
/// produce coefficients in `{−6, −3, −1, 0, 1, 3, 6}` worst-case; a
/// bound of `1e6` catches any genuine numerical blowup well before
/// `f64` precision loss.
pub const COEFF_BOUND: f64 = 1.0e6;

/// Errors produced by the Gröbner machinery.
#[derive(Debug, Clone)]
pub enum GroebnerError {
    /// A coefficient grew beyond [`COEFF_BOUND`]; this signals either
    /// a non-trivial denominator pattern unsupported by the `f64`
    /// representation or a bug in the reduction.
    CoefficientBlowup(f64),
    /// Division by a basis element with leading coefficient 0 (should
    /// never happen if the basis is constructed by `reduced_groebner`).
    LeadingCoefficientZero,
}

impl std::fmt::Display for GroebnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CoefficientBlowup(c) => write!(
                f,
                "groebner: coefficient blowup |c| = {c} exceeds bound {COEFF_BOUND}"
            ),
            Self::LeadingCoefficientZero => {
                write!(f, "groebner: divisor has zero leading coefficient")
            }
        }
    }
}

impl std::error::Error for GroebnerError {}

/// Monomial-ordering kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderKind {
    /// Pure lexicographic.
    Lex,
    /// Total degree first, ties broken by lex.
    DegLex,
    /// Total degree first, ties broken by reverse lex.
    DegRevLex,
}

/// Monomial order on [`Exponent`].
#[derive(Debug, Clone, Copy)]
pub struct MonomialOrder {
    pub kind: OrderKind,
}

impl MonomialOrder {
    pub fn new(kind: OrderKind) -> Self {
        Self { kind }
    }

    /// `cmp(a, b)` returns `Ordering::Greater` iff `a > b` in the order.
    #[inline]
    pub fn cmp(&self, a: &Exponent, b: &Exponent) -> Ordering {
        match self.kind {
            OrderKind::Lex => lex_cmp(a, b),
            OrderKind::DegLex => {
                let da: u32 = a.iter().sum();
                let db: u32 = b.iter().sum();
                match da.cmp(&db) {
                    Ordering::Equal => lex_cmp(a, b),
                    o => o,
                }
            }
            OrderKind::DegRevLex => {
                let da: u32 = a.iter().sum();
                let db: u32 = b.iter().sum();
                match da.cmp(&db) {
                    Ordering::Equal => revlex_cmp(a, b),
                    o => o,
                }
            }
        }
    }
}

#[inline]
fn lex_cmp(a: &Exponent, b: &Exponent) -> Ordering {
    for i in 0..NVAR {
        match a[i].cmp(&b[i]) {
            Ordering::Equal => continue,
            o => return o,
        }
    }
    Ordering::Equal
}

#[inline]
fn revlex_cmp(a: &Exponent, b: &Exponent) -> Ordering {
    // Reverse-lex: scan from the LAST variable; smaller "last" exponent
    // is GREATER (Cox-Little-O'Shea Def. 2.2.6).
    for i in (0..NVAR).rev() {
        match a[i].cmp(&b[i]) {
            Ordering::Equal => continue,
            // Smaller a[i] => a is greater.
            o => return o.reverse(),
        }
    }
    Ordering::Equal
}

/// Coefficient type. `f64` is exact on the small-integer values that
/// arise in the chapter-21 CICY problems.
pub type Coeff = f64;

/// Polynomial: sorted-descending list of `(coeff, exponent)` terms
/// (sorted by `order`), zeros removed.
#[derive(Debug, Clone)]
pub struct Polynomial {
    pub terms: Vec<(Coeff, Exponent)>,
    pub order: MonomialOrder,
}

impl Polynomial {
    /// Construct from a list of `(coeff, exponent)` pairs in any order.
    /// Duplicates are collected, zeros pruned, and the result sorted in
    /// descending monomial order.
    pub fn from_terms(mut terms: Vec<(Coeff, Exponent)>, order: MonomialOrder) -> Self {
        // Sort by exponent (ascending lex by raw bytes for grouping).
        terms.sort_by(|a, b| {
            for i in 0..NVAR {
                match a.1[i].cmp(&b.1[i]) {
                    Ordering::Equal => continue,
                    o => return o,
                }
            }
            Ordering::Equal
        });
        // Collapse duplicates.
        let mut collapsed: Vec<(Coeff, Exponent)> = Vec::with_capacity(terms.len());
        for (c, e) in terms {
            if let Some(last) = collapsed.last_mut() {
                if last.1 == e {
                    last.0 += c;
                    continue;
                }
            }
            collapsed.push((c, e));
        }
        // Drop zeros.
        collapsed.retain(|(c, _)| c.abs() > ZERO_TOL);
        // Sort in descending monomial order.
        collapsed.sort_by(|a, b| order.cmp(&b.1, &a.1));
        Self {
            terms: collapsed,
            order,
        }
    }

    /// The zero polynomial.
    pub fn zero(order: MonomialOrder) -> Self {
        Self {
            terms: Vec::new(),
            order,
        }
    }

    /// Construct a single-term polynomial from `(coeff, exponent)`.
    pub fn monomial(coeff: Coeff, exponent: Exponent, order: MonomialOrder) -> Self {
        if coeff.abs() <= ZERO_TOL {
            return Self::zero(order);
        }
        Self {
            terms: vec![(coeff, exponent)],
            order,
        }
    }

    /// Is this the zero polynomial?
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Leading coefficient (`0.0` for the zero polynomial).
    #[inline]
    pub fn lc(&self) -> Coeff {
        self.terms.first().map(|(c, _)| *c).unwrap_or(0.0)
    }

    /// Leading monomial (`[0; NVAR]` for the zero polynomial).
    #[inline]
    pub fn lm(&self) -> Exponent {
        self.terms.first().map(|(_, e)| *e).unwrap_or([0; NVAR])
    }

    /// Number of (nonzero) terms.
    #[inline]
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// In-place scalar multiply.
    pub fn scale(&mut self, s: Coeff) {
        if s.abs() <= ZERO_TOL {
            self.terms.clear();
            return;
        }
        for (c, _) in self.terms.iter_mut() {
            *c *= s;
        }
    }

    /// Multiply by a single monomial `c * x^e` (in place).
    pub fn mul_monomial(&mut self, c: Coeff, e: &Exponent) {
        if c.abs() <= ZERO_TOL {
            self.terms.clear();
            return;
        }
        for (cc, ee) in self.terms.iter_mut() {
            *cc *= c;
            for k in 0..NVAR {
                ee[k] += e[k];
            }
        }
        // Order is preserved under multiplication by a monomial in any
        // admissible order.
    }
}

/// Add two polynomials.
pub fn poly_add(a: &Polynomial, b: &Polynomial) -> Polynomial {
    debug_assert!(a.order.kind == b.order.kind);
    let mut terms = Vec::with_capacity(a.terms.len() + b.terms.len());
    terms.extend_from_slice(&a.terms);
    terms.extend_from_slice(&b.terms);
    Polynomial::from_terms(terms, a.order)
}

/// Subtract `b` from `a`.
pub fn poly_sub(a: &Polynomial, b: &Polynomial) -> Polynomial {
    debug_assert!(a.order.kind == b.order.kind);
    let mut terms = Vec::with_capacity(a.terms.len() + b.terms.len());
    terms.extend_from_slice(&a.terms);
    for (c, e) in &b.terms {
        terms.push((-*c, *e));
    }
    Polynomial::from_terms(terms, a.order)
}

/// Multiply two polynomials.
pub fn poly_mul(a: &Polynomial, b: &Polynomial) -> Polynomial {
    debug_assert!(a.order.kind == b.order.kind);
    let mut terms = Vec::with_capacity(a.terms.len() * b.terms.len());
    for (ca, ea) in &a.terms {
        for (cb, eb) in &b.terms {
            let mut e = [0u32; NVAR];
            for k in 0..NVAR {
                e[k] = ea[k] + eb[k];
            }
            terms.push((*ca * *cb, e));
        }
    }
    Polynomial::from_terms(terms, a.order)
}

/// LCM of two exponent vectors (component-wise max).
#[inline]
fn lcm_exp(a: &Exponent, b: &Exponent) -> Exponent {
    let mut out = [0u32; NVAR];
    for k in 0..NVAR {
        out[k] = a[k].max(b[k]);
    }
    out
}

/// Component-wise difference; returns `None` if any component is
/// negative (i.e. `a` does not divide `b`).
#[inline]
fn try_sub_exp(b: &Exponent, a: &Exponent) -> Option<Exponent> {
    let mut out = [0u32; NVAR];
    for k in 0..NVAR {
        if a[k] > b[k] {
            return None;
        }
        out[k] = b[k] - a[k];
    }
    Some(out)
}

/// `LM(f) | LM(g)` ?
#[inline]
fn lm_divides(divisor: &Polynomial, target: &Polynomial) -> bool {
    if divisor.is_zero() || target.is_zero() {
        return false;
    }
    let lm_d = divisor.lm();
    let lm_t = target.lm();
    for k in 0..NVAR {
        if lm_d[k] > lm_t[k] {
            return false;
        }
    }
    true
}

/// Buchberger's S-polynomial:
///
///   `S(f, g) = (lcm / LT(f)) · f − (lcm / LT(g)) · g`
///
/// where `lcm = lcm(LM(f), LM(g))` and `LT(·)` is the leading term.
/// (Cox-Little-O'Shea Def. 2.6.4.)
pub fn s_polynomial(f: &Polynomial, g: &Polynomial) -> Polynomial {
    debug_assert!(f.order.kind == g.order.kind);
    if f.is_zero() || g.is_zero() {
        return Polynomial::zero(f.order);
    }
    let lm_f = f.lm();
    let lm_g = g.lm();
    let lc_f = f.lc();
    let lc_g = g.lc();
    let lcm = lcm_exp(&lm_f, &lm_g);
    let alpha_f = try_sub_exp(&lcm, &lm_f).expect("lcm must contain LM(f)");
    let alpha_g = try_sub_exp(&lcm, &lm_g).expect("lcm must contain LM(g)");
    // S = (1/lc(f)) x^alpha_f * f  -  (1/lc(g)) x^alpha_g * g
    let mut left = f.clone();
    left.mul_monomial(1.0 / lc_f, &alpha_f);
    let mut right = g.clone();
    right.mul_monomial(1.0 / lc_g, &alpha_g);
    poly_sub(&left, &right)
}

/// Multivariate polynomial division (Cox-Little-O'Shea §2.3 Thm 3):
///
///   `f = Σ q_i g_i + r`
///
/// where no monomial of `r` is divisible by any `LM(g_i)`. Returns
/// `(q, r)`.
pub fn divide_with_remainder(
    f: &Polynomial,
    basis: &[Polynomial],
) -> Result<(Vec<Polynomial>, Polynomial), GroebnerError> {
    let order = f.order;
    let mut p = f.clone();
    let mut quotients: Vec<Polynomial> = (0..basis.len())
        .map(|_| Polynomial::zero(order))
        .collect();
    let mut remainder = Polynomial::zero(order);

    while !p.is_zero() {
        // Find a basis element whose LM divides LM(p).
        let lm_p = p.lm();
        let lc_p = p.lc();
        let mut divided = false;
        for (i, g) in basis.iter().enumerate() {
            if g.is_zero() {
                continue;
            }
            if let Some(alpha) = try_sub_exp(&lm_p, &g.lm()) {
                let lc_g = g.lc();
                if lc_g.abs() < ZERO_TOL {
                    return Err(GroebnerError::LeadingCoefficientZero);
                }
                let factor = lc_p / lc_g;
                if !factor.is_finite() || factor.abs() > COEFF_BOUND {
                    return Err(GroebnerError::CoefficientBlowup(factor));
                }
                // Update quotient i: q_i += factor * x^alpha.
                let mono = Polynomial::monomial(factor, alpha, order);
                quotients[i] = poly_add(&quotients[i], &mono);
                // Update p: p -= factor * x^alpha * g_i.
                let mut sub_term = g.clone();
                sub_term.mul_monomial(factor, &alpha);
                p = poly_sub(&p, &sub_term);
                divided = true;
                break;
            }
        }
        if !divided {
            // Move LT(p) to remainder.
            let mono = Polynomial::monomial(lc_p, lm_p, order);
            remainder = poly_add(&remainder, &mono);
            // p = p - LT(p).
            p.terms.remove(0);
        }
        // Sanity guard.
        if let Some((c, _)) = p.terms.iter().max_by(|a, b| a.0.abs().total_cmp(&b.0.abs())) {
            if !c.is_finite() || c.abs() > COEFF_BOUND {
                return Err(GroebnerError::CoefficientBlowup(*c));
            }
        }
    }

    Ok((quotients, remainder))
}

/// Buchberger's algorithm with the standard *product criterion* +
/// LCM-based S-pair filtering.
///
/// The product criterion (Cox-Little-O'Shea §2.9 Prop. 4):
/// if `gcd(LM(f), LM(g)) = 1` (i.e. `lcm = LM(f) · LM(g)`), then
/// `S(f, g)` reduces to zero modulo `{f, g}`, so the pair can be
/// skipped.
///
/// S-pairs are reduced in parallel via rayon over independent
/// computations; their non-zero remainders are then folded back into
/// the basis sequentially (the basis itself is the synchronisation
/// point).
pub fn buchberger(generators: Vec<Polynomial>) -> Result<Vec<Polynomial>, GroebnerError> {
    if generators.is_empty() {
        return Ok(Vec::new());
    }
    let order = generators[0].order;
    let mut g: Vec<Polynomial> = generators.into_iter().filter(|p| !p.is_zero()).collect();
    if g.is_empty() {
        return Ok(Vec::new());
    }

    // Worklist of pairs `(i, j)` with i < j to process.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..g.len() {
        for j in (i + 1)..g.len() {
            pairs.push((i, j));
        }
    }

    while !pairs.is_empty() {
        // Filter by product criterion: skip pairs with disjoint LMs.
        let active: Vec<(usize, usize)> = pairs
            .drain(..)
            .filter(|(i, j)| {
                let li = g[*i].lm();
                let lj = g[*j].lm();
                let mut disjoint = true;
                for k in 0..NVAR {
                    if li[k] > 0 && lj[k] > 0 {
                        disjoint = false;
                        break;
                    }
                }
                !disjoint
            })
            .collect();

        if active.is_empty() {
            break;
        }

        // Reduce S-polynomials in parallel; collect non-zero remainders.
        let g_snapshot = g.clone();
        let new_polys: Vec<Polynomial> = active
            .par_iter()
            .map(|(i, j)| {
                let s = s_polynomial(&g_snapshot[*i], &g_snapshot[*j]);
                if s.is_zero() {
                    return Ok(Polynomial::zero(order));
                }
                let (_, r) = divide_with_remainder(&s, &g_snapshot)?;
                Ok(r)
            })
            .collect::<Result<Vec<_>, GroebnerError>>()?;

        // Fold non-zero remainders back. For each addition, generate the
        // new pairs that need processing.
        for r in new_polys {
            if r.is_zero() {
                continue;
            }
            // Avoid trivial duplicates: if some element's LM equals the
            // new LM and the new element reduces to zero modulo current
            // basis, skip. This is an optimisation; soundness does not
            // depend on it.
            let new_idx = g.len();
            for k in 0..new_idx {
                pairs.push((k, new_idx));
            }
            g.push(r);
        }
    }

    Ok(g)
}

/// Reduced minimal Gröbner basis (Cox-Little-O'Shea §2.7 Thm. 5).
///
/// 1. Run Buchberger to get a Gröbner basis.
/// 2. Drop `g_i` if `LM(g_i)` is divisible by `LM(g_j)` for some `j ≠ i`
///    (minimality).
/// 3. Reduce each surviving `g_i` modulo the others (so no monomial of
///    `g_i` is divisible by any other `LM(g_j)`).
/// 4. Normalise leading coefficients to `1`.
///
/// The result is unique modulo the chosen monomial order (Cox-Little-
/// O'Shea Cor. 2.7.6).
pub fn reduced_groebner(generators: Vec<Polynomial>) -> Result<Vec<Polynomial>, GroebnerError> {
    let g = buchberger(generators)?;
    if g.is_empty() {
        return Ok(g);
    }

    // (2) Drop redundant generators.
    let mut keep: Vec<bool> = vec![true; g.len()];
    for i in 0..g.len() {
        if !keep[i] {
            continue;
        }
        let lm_i = g[i].lm();
        for j in 0..g.len() {
            if i == j || !keep[j] {
                continue;
            }
            let lm_j = g[j].lm();
            // Strict divisibility: LM(j) | LM(i) and LM(j) != LM(i).
            let mut divides = true;
            let mut equal = true;
            for k in 0..NVAR {
                if lm_j[k] > lm_i[k] {
                    divides = false;
                    break;
                }
                if lm_j[k] != lm_i[k] {
                    equal = false;
                }
            }
            if divides && !equal {
                keep[i] = false;
                break;
            }
        }
    }
    let mut minimal: Vec<Polynomial> = g
        .into_iter()
        .zip(keep.into_iter())
        .filter_map(|(p, k)| if k { Some(p) } else { None })
        .collect();

    // De-duplicate by leading monomial: among polynomials sharing a LM,
    // keep only one (uniqueness up to scalar).
    minimal.sort_by(|a, b| {
        let oa = a.order;
        oa.cmp(&b.lm(), &a.lm())
    });
    let mut dedup: Vec<Polynomial> = Vec::with_capacity(minimal.len());
    for p in minimal {
        if let Some(last) = dedup.last() {
            if last.lm() == p.lm() {
                continue;
            }
        }
        dedup.push(p);
    }
    let mut minimal = dedup;

    // (3) Inter-reduce.
    let mut changed = true;
    let mut guard = 0usize;
    while changed && guard < 64 {
        guard += 1;
        changed = false;
        for i in 0..minimal.len() {
            // Reduce minimal[i] modulo all others.
            let mut others: Vec<Polynomial> = Vec::with_capacity(minimal.len() - 1);
            for j in 0..minimal.len() {
                if i != j {
                    others.push(minimal[j].clone());
                }
            }
            let (_, r) = divide_with_remainder(&minimal[i], &others)?;
            // Compare term lists; if they differ, replace.
            if r.terms != minimal[i].terms {
                minimal[i] = r;
                changed = true;
            }
        }
        // Drop any polynomials reduced to zero.
        minimal.retain(|p| !p.is_zero());
    }

    // (4) Normalise leading coefficients to 1.
    for p in minimal.iter_mut() {
        let lc = p.lc();
        if lc.abs() < ZERO_TOL {
            return Err(GroebnerError::LeadingCoefficientZero);
        }
        p.scale(1.0 / lc);
    }

    // Sort by descending leading monomial for deterministic output.
    minimal.sort_by(|a, b| a.order.cmp(&b.lm(), &a.lm()));

    Ok(minimal)
}

// ---------------------------------------------------------------------------
// Convenience constructors for the route34 ideals
// ---------------------------------------------------------------------------

/// Build the TY defining ideal generators in `8` variables ordered
/// `[z_0, z_1, z_2, z_3, w_0, w_1, w_2, w_3]`:
///
///   `f_1 = z_0³ + z_1³ + z_2³ + z_3³`,
///   `f_2 = w_0³ + w_1³ + w_2³ + w_3³`,
///   `f_3 = Σ_i z_i w_i`.
pub fn ty_generators(order: MonomialOrder) -> Vec<Polynomial> {
    let mut f1_terms: Vec<(Coeff, Exponent)> = Vec::with_capacity(4);
    for k in 0..4 {
        let mut e = [0u32; NVAR];
        e[k] = 3;
        f1_terms.push((1.0, e));
    }
    let mut f2_terms: Vec<(Coeff, Exponent)> = Vec::with_capacity(4);
    for k in 0..4 {
        let mut e = [0u32; NVAR];
        e[4 + k] = 3;
        f2_terms.push((1.0, e));
    }
    let mut f3_terms: Vec<(Coeff, Exponent)> = Vec::with_capacity(4);
    for k in 0..4 {
        let mut e = [0u32; NVAR];
        e[k] = 1;
        e[4 + k] = 1;
        f3_terms.push((1.0, e));
    }
    vec![
        Polynomial::from_terms(f1_terms, order),
        Polynomial::from_terms(f2_terms, order),
        Polynomial::from_terms(f3_terms, order),
    ]
}

/// Build the Schoen defining ideal generators in `8` variables ordered
/// `[x_0, x_1, x_2, y_0, y_1, y_2, t_0, t_1]`. The ideal is generated
/// by two generic polynomials of bidegree `(3, 0, 1)` and `(0, 3, 1)`:
///
///   `F_1 = (x_0³ + x_1³ + x_2³) t_0 + (x_0³ − x_1³ − x_2³) t_1`,
///   `F_2 = (y_0³ + y_1³ + y_2³) t_0 + (y_0³ − y_1³ − y_2³) t_1`.
///
/// (The specific cubic combinations are the `Z/3 × Z/3`-invariant
/// representatives used by [`crate::route34::schoen_sampler::SchoenPoly::
/// z3xz3_invariant_default`]; see `schoen_sampler.rs` for the canonical
/// definition. We only need the monomial-leading-term structure here,
/// so any non-degenerate generic representative produces the same
/// reduced Gröbner basis at the level of leading-monomial counts.)
pub fn schoen_generators(order: MonomialOrder) -> Vec<Polynomial> {
    // F_1: in variables (x_0..x_2, t_0, t_1) at indices (0,1,2,6,7).
    let mut f1_terms: Vec<(Coeff, Exponent)> = Vec::with_capacity(6);
    for k in 0..3 {
        // x_k^3 t_0
        let mut e = [0u32; NVAR];
        e[k] = 3;
        e[6] = 1;
        f1_terms.push((1.0, e));
        // sign pattern: (+ - -) on t_1 block.
        let mut e2 = [0u32; NVAR];
        e2[k] = 3;
        e2[7] = 1;
        let sign = if k == 0 { 1.0 } else { -1.0 };
        f1_terms.push((sign, e2));
    }
    // F_2: in variables (y_0..y_2, t_0, t_1) at indices (3,4,5,6,7).
    let mut f2_terms: Vec<(Coeff, Exponent)> = Vec::with_capacity(6);
    for k in 0..3 {
        let mut e = [0u32; NVAR];
        e[3 + k] = 3;
        e[6] = 1;
        f2_terms.push((1.0, e));
        let mut e2 = [0u32; NVAR];
        e2[3 + k] = 3;
        e2[7] = 1;
        let sign = if k == 0 { 1.0 } else { -1.0 };
        f2_terms.push((sign, e2));
    }
    vec![
        Polynomial::from_terms(f1_terms, order),
        Polynomial::from_terms(f2_terms, order),
    ]
}

/// Test whether the monomial `m` is in the ideal generated by `basis`,
/// i.e. whether `divide_with_remainder(m, basis).remainder == 0`.
///
/// Convenience wrapper used by `ty_metric.rs` /
/// `schoen_metric.rs::build_*_invariant_reduced_basis` to filter
/// monomials that reduce to zero modulo the (reduced) Gröbner basis.
pub fn monomial_in_ideal(
    m: &Exponent,
    basis: &[Polynomial],
    order: MonomialOrder,
) -> Result<bool, GroebnerError> {
    let f = Polynomial::monomial(1.0, *m, order);
    let (_, r) = divide_with_remainder(&f, basis)?;
    Ok(r.is_zero())
}

/// Test whether the monomial `m` lies in the *monomial ideal* generated
/// by `{LM(g) : g ∈ basis}` — equivalently, whether some leading
/// monomial of the Gröbner basis divides `m`.
///
/// By Cox-Little-O'Shea §2.7 Thm 5, the set of monomials NOT in this
/// monomial ideal is a vector-space basis for the quotient `R / I`. So
/// a section-basis monomial that DOES lie in the LM-ideal is redundant
/// modulo the defining ideal and must be dropped from the Donaldson
/// section basis.
///
/// This is the operation used by `build_ty_invariant_reduced_basis` /
/// `build_schoen_invariant_reduced_basis` to do real Buchberger
/// normal-form filtering.
pub fn monomial_in_lm_ideal(m: &Exponent, basis: &[Polynomial]) -> bool {
    for g in basis {
        if g.is_zero() {
            continue;
        }
        let lm = g.lm();
        let mut divides = true;
        for k in 0..NVAR {
            if lm[k] > m[k] {
                divides = false;
                break;
            }
        }
        if divides {
            return true;
        }
    }
    false
}
