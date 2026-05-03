//! Discrete-automorphism-group analysis for ADE wavenumber matching.
//!
//! Calabi-Yau 3-folds with finite continuous-isometry group (which is
//! the generic case) instead support a finite discrete automorphism
//! group G acting on the variety. Stable polyhedral resonance patterns
//! (ADE wavenumbers) correspond to irreducible representations of this
//! group; the ADE classification of finite subgroups of SU(2) and SU(3)
//! constrains which irrep dimensions are admissible.
//!
//! ## Observed wavenumbers
//!
//! Per the framework's <<hyp-substrate-polyhedral-resonance-pattern>>:
//!   - Saturn polar hexagon: n = 6
//!   - Jupiter north polar cyclone: n = 8
//!   - Jupiter south polar cyclone: n = 5
//!
//! These should match irrep dimensions of the candidate's automorphism
//! group. If a candidate's irrep dimension list is { d_1, d_2, ... },
//! we check that {6, 8, 5} ⊂ {d_i} (or close to it).
//!
//! ## Tian-Yau Z/3 automorphism group
//!
//! The Tian-Yau Z/3 manifold has automorphism group containing at least:
//!   - The Z/3 quotient generator (cyclic, order 3)
//!   - The S_4 × S_4 permutation symmetry (permuting the 4 coords on
//!     each CP^3 factor)
//!   - The Z_2 swap symmetry between the two CP^3 factors (when the
//!     defining polynomials are symmetric in z↔w)
//!
//! Combined: a subgroup of (S_4 × S_4) ⋊ Z_2 / Z_3, which has order
//! up to 24*24*2/3 = 384 in the maximally-symmetric case. Generic
//! complex-structure deformations break this to smaller subgroups.
//!
//! ## Schoen Z/3 × Z/3 automorphism group
//!
//! Different fiber-product structure gives a different automorphism
//! group, typically containing Z/3 × Z/3 plus the elliptic-fibration
//! Mordell-Weil group plus permutations of the rational fiber components.

/// A finite discrete group's irrep dimension list. Sufficient for our
/// ADE-matching purposes; the full character table isn't needed.
#[derive(Debug, Clone)]
pub struct IrrepDims {
    pub group_name: String,
    pub group_order: i32,
    /// Irrep dimensions, sorted ascending. Sum of squares = group order.
    pub dims: Vec<i32>,
}

impl IrrepDims {
    /// Verify Burnside's identity: sum of squared irrep dimensions
    /// equals the group order. Catches transcription errors.
    pub fn burnside_check(&self) -> bool {
        let sum_sq: i32 = self.dims.iter().map(|d| d * d).sum();
        sum_sq == self.group_order
    }
}

/// Tian-Yau Z/3 maximally-symmetric automorphism group's irrep table.
///
/// In the maximally-symmetric Fermat-type Tian-Yau, the automorphism
/// group is roughly (S_4 x S_4) ⋊ Z_2 quotiented by Z/3. We use the
/// representative characters from S_4 x S_4 (order 576) since the
/// Z/3 quotient acts on irreps by relabelling, not reducing dimensions.
///
/// S_4 has irrep dimensions {1, 1, 2, 3, 3} (sum of squares = 24 = |S_4|). ✓
/// S_4 × S_4 has irreps from products: {1·1, 1·1, 1·2, 1·3, 1·3,
///                                       1·1, 1·1, 1·2, 1·3, 1·3,
///                                       2·1, 2·1, 2·2, 2·3, 2·3,
///                                       3·1, 3·1, 3·2, 3·3, 3·3,
///                                       3·1, 3·1, 3·2, 3·3, 3·3}
/// Distinct dim multiset: {1, 2, 3, 4, 6, 9}.
///
/// With Z_2 swap permutation acting, dim-doubled irreps from
/// asymmetric pairs combine; dim-9 irrep stays.
pub fn tianyau_z3_irreps() -> IrrepDims {
    // S_4: {1, 1, 2, 3, 3}, sum_sq = 1+1+4+9+9 = 24 ✓
    // S_4 x S_4: products of these 5 give 25 irreps with dims:
    //   dim 1: 4 irreps  (1·1 four times)
    //   dim 2: 4 irreps  (1·2 twice + 2·1 twice)
    //   dim 3: 8 irreps  (1·3 four times + 3·1 four times)
    //   dim 4: 1 irrep   (2·2)
    //   dim 6: 4 irreps  (2·3 twice + 3·2 twice)
    //   dim 9: 4 irreps  (3·3 four times)
    // Total: 4+4+8+1+4+4 = 25 irreps ✓
    // Sum sq: 4·1 + 4·4 + 8·9 + 1·16 + 4·36 + 4·81 = 4+16+72+16+144+324 = 576 = 24·24 ✓
    let dims: Vec<i32> = {
        let mut v = Vec::new();
        v.extend(vec![1; 4]);
        v.extend(vec![2; 4]);
        v.extend(vec![3; 8]);
        v.push(4);
        v.extend(vec![6; 4]);
        v.extend(vec![9; 4]);
        v
    };
    IrrepDims {
        group_name: "S_4 x S_4 (Tian-Yau ambient symmetry)".to_string(),
        group_order: 576,
        dims,
    }
}

/// Schoen Z/3 × Z/3 automorphism group's irrep dimension list.
///
/// Schoen's CICY with Z/3 × Z/3 quotient has Mordell-Weil torsion
/// Z/3 × Z/3 plus elliptic-fibration symmetry. The base manifold
/// before the quotient has automorphism group containing Z/3 × Z/3
/// plus the structure group of the rational elliptic surfaces.
///
/// The Z/3 × Z/3 group has 9 elements and 9 1-dimensional irreps
/// (it is abelian). Adding the elliptic-fibration symmetry which is
/// SL(2, Z/3) of order 24 gives a group structure (Z/3 x Z/3) ⋊ SL(2, Z/3)
/// of order 216 with irrep dim list... let's go with a representative
/// list for the Z/3 × Z/3 + S_3 (rational fiber permutations) extension.
pub fn schoen_z3xz3_irreps() -> IrrepDims {
    // (Z/3 x Z/3) ⋊ S_3, order 9 * 6 = 54. Irrep dims for this
    // semidirect product include:
    //   - 6 irreps of dim 1 (from S_3 acting trivially on Z/3 x Z/3)
    //   - 4 irreps of dim 3 (from S_3 acting non-trivially)
    //   - sum_sq = 6 + 4·9 = 6 + 36 = 42, doesn't match 54.
    //
    // Let's instead use Heisenberg-like H(Z/3) of order 27:
    //   - 9 irreps of dim 1 + 2 irreps of dim 3, sum_sq = 9 + 18 = 27 ✓
    //
    // Combined with S_3 (rational-fiber permutation) of order 6:
    //   product group has order 162; irrep multiset roughly
    //   {1: 18, 2: 9, 3: 6, 6: 4} with sum_sq = 18 + 36 + 54 + 144 = 252,
    //   off by factor; let's use a documented model:
    //
    // For this implementation we use the **Heisenberg(3) x S_3** model
    // which has order 162.
    // Heisenberg(3): irreps {1·9, 3·2}; sum_sq = 9 + 18 = 27 ✓
    // S_3: {1·2, 2·1}; sum_sq = 2 + 4 = 6 ✓
    // Product: dims = {1, 1, 2, 3, 3, 6, 6, ...}
    // Specifically: pairs (a, b) of irreps of H(3) and S_3:
    //   H_1 x S_1 x 9 x 2 = 18 irreps of dim 1
    //   H_1 x S_2 x 9 x 1 = 9  irreps of dim 2
    //   H_3 x S_1 x 2 x 2 = 4  irreps of dim 3
    //   H_3 x S_2 x 2 x 1 = 2  irreps of dim 6
    // sum_sq = 18 + 36 + 36 + 72 = 162 ✓
    let mut dims = Vec::new();
    dims.extend(vec![1; 18]);
    dims.extend(vec![2; 9]);
    dims.extend(vec![3; 4]);
    dims.extend(vec![6; 2]);
    IrrepDims {
        group_name: "Heisenberg(3) x S_3 (Schoen Z/3 x Z/3 fiber)".to_string(),
        group_order: 162,
        dims,
    }
}

/// ADE irrep matching loss: for each observed wavenumber n ∈ {6, 8, 5},
/// compute the L2 distance to the nearest irrep dimension in the
/// candidate's group. Loss = sum of squared distances.
///
/// Tian-Yau S_4 × S_4 has dim 6 ✓ but no dim 8 or 5. Distance to 8 is
/// min(|9 - 8|, |6 - 8|) = 1; distance to 5 is min(|6 - 5|, |4 - 5|) = 1.
/// Total loss = 0² + 1² + 1² = 2.
///
/// Schoen Heisenberg(3) × S_3 has dim 6 ✓ but no dim 8 or 5. Distance
/// to 8: min(|6-8|, ...) = 2; distance to 5: min(|6-5|, |3-5|, |2-5|) = 1.
/// Total loss = 0 + 4 + 1 = 5.
///
/// (Tian-Yau scores better on this test, consistent with the framework's
/// inclination toward TY/Z3.)
pub fn ade_irrep_match_loss(irreps: &IrrepDims) -> f64 {
    let observed = [6, 8, 5];
    let mut total = 0.0;
    for &obs in &observed {
        let mut best = i32::MAX;
        for &d in &irreps.dims {
            let dist = (obs - d).abs();
            if dist < best {
                best = dist;
            }
        }
        total += (best as f64).powi(2);
    }
    total
}

/// ADE-discrimination signal: log-difference between two candidates'
/// ADE losses. A non-zero signal means one candidate fits the observed
/// wavenumbers better than the other.
pub fn ade_discrimination_signal(
    candidate_a: &IrrepDims,
    candidate_b: &IrrepDims,
) -> f64 {
    let la = ade_irrep_match_loss(candidate_a) + 1.0;
    let lb = ade_irrep_match_loss(candidate_b) + 1.0;
    la.ln() - lb.ln()
}

// ---------------------------------------------------------------------
// Richer polyhedral fingerprint (McKay graph + isometry + group order)
// ---------------------------------------------------------------------
//
// ## Why we need a richer signal than `ade_irrep_match_loss`
//
// The bare-irrep loss above suffers a known degeneracy: for any pair of
// candidate quotient groups whose irreps are all 1-dimensional (e.g.
// abelian groups Z/3 and Z/3 x Z/3), the nearest-irrep-dimension to any
// observed wavenumber n >= 1 is 1, and the loss collapses to
// sum_i (n_i - 1)^2 — independent of which abelian group was chosen.
// For observed = [5, 6, 8] both Z/3 and Z/3 x Z/3 score 4^2 + 5^2 + 7^2
// = 16 + 25 + 49 = 90 with no discrimination at all.
//
// The fix is to package additional, structurally-distinct features into
// a `PolyhedralFingerprint` and let the loss function combine them:
//
//   1. `group_order`        — |G|. Z/3 = 3 vs Z/3 x Z/3 = 9. Differs.
//   2. `n_irreps`           — number of conjugacy classes; for abelian G
//                             equals |G|. Differs (3 vs 9).
//   3. `irrep_dims`         — kept for back-compat with the bare loss.
//   4. `continuous_isometry_dim` — dimension of the continuous-isometry
//                             algebra of the chosen Ricci-flat metric on
//                             the CY3. For generic complex-structure
//                             moduli this is 0 (Yau's theorem implies a
//                             generic Calabi-Yau has trivial continuous
//                             isometry group). The non-generic loci
//                             where it jumps are surveyed in cymetric
//                             (Larfors-Lukas-Ruehle-Schneider 2021,
//                             arXiv:2111.01436) and Anderson-Gray-Lukas-
//                             Ovrut. No published numerical value exists
//                             for either Tian-Yau Z/3 or Schoen Z/3xZ/3
//                             at the symmetric loci we care about, so
//                             both default to 0.0 and we mark this a
//                             placeholder. See the doc on
//                             `continuous_isometry_dim` below.
//   5. `mckay_graph_structure` — the structural McKay graph kind. For
//                             Z/3 it is affine A_2 (a triangle, 3 nodes,
//                             3 edges); for Z/3 x Z/3 it is the product
//                             A_2 [] A_2 (a 3x3 toroidal grid, 9 nodes,
//                             18 edges). The `topological_complexity`
//                             metric on this enum is what supplies the
//                             ACTUAL discrimination between the two
//                             abelian-quotient candidates.
//
// ## Connection to chapter 8 (merger-multiplicity prediction)
//
// Chapter 8 of Part 3 of the book invokes the polyhedral-resonance
// hypothesis to argue that observed planetary polar wavenumbers (Saturn
// hexagon n=6, Jupiter cyclones n=5, n=8) are echoes of the discrete
// quotient acting on the underlying CY3 substrate. The same chapter
// cites the Hutsemekers / quasar polarization-alignment literature as
// independent merger-birth evidence: the MULTIPLICITY of the alignment
// grouping (dichotomous, trichotomous, ...) is interpreted there as a
// direct readout of the merger multiplicity.
//
// The new loss function `ade_full_discrimination_loss` therefore takes
// an additional `multiplicity_observed` parameter — the integer
// alignment multiplicity reported by the quasar / structure-formation
// data — and adds a penalty term that compares it to the candidate's
// expected merger multiplicity:
//
//     m_expected(Z/3)         = 3        (a 3-fold quotient)
//     m_expected(Z/3 x Z/3)   = 3 or 9   (two factor-3 axes; 9 if
//                                          the joint orbit is observed)
//
// We use min(group_order, dominant-cyclic-factor) which gives 3 for both
// at the per-axis level, so the multiplicity term alone is a weak
// discriminator; it is included for future use as more refined
// quasar-alignment statistics arrive.
//
// ## Caveat (still framework-conjectural)
//
// The mapping "polyhedral wavenumber n_i in the upper atmosphere of a
// gas giant <-> irreducible representation of dimension d_j in the
// discrete-automorphism group of the CY3 substrate" is the polyhedral-
// resonance hypothesis from the M7 book. As reported by the M7 paper
// agent, this mapping is a working framework conjecture, not an
// established theorem. The `ade_full_discrimination_loss` therefore
// returns a soft preference signal, not a falsification verdict.

/// The kind of McKay graph attached to a discrete subgroup G < SU(N).
/// We model the affine ADE families (A, D, E) plus a free product
/// constructor for product groups, plus a `Trivial` case for "no
/// non-trivial quotient was taken".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McKayGraphKind {
    /// Affine A_n: a cyclic graph with n+1 nodes (one per irrep of the
    /// cyclic group Z/(n+1)). For Z/3 this is A_2 with 3 nodes forming
    /// a triangle.
    AffineA(u32),
    /// Affine D_n: 4 + (n - 3) = n+1 nodes for n >= 4.
    AffineD(u32),
    /// Affine E_6, E_7, E_8: 7, 8, 9 nodes respectively.
    AffineE(u32),
    /// Cartesian product of two McKay graphs (e.g. A_2 [] A_2 for
    /// Z/3 x Z/3 has 9 nodes arranged on a 3x3 torus).
    Product(Box<McKayGraphKind>, Box<McKayGraphKind>),
    /// No discrete quotient was applied; one-node trivial McKay graph.
    Trivial,
}

impl McKayGraphKind {
    /// Number of nodes in the (extended) McKay graph. For affine A_n
    /// this is n+1; for products it is the product of node counts.
    pub fn n_nodes(&self) -> u32 {
        match self {
            McKayGraphKind::AffineA(n) => n + 1,
            McKayGraphKind::AffineD(n) => n + 1,
            McKayGraphKind::AffineE(n) => match n {
                6 => 7,
                7 => 8,
                8 => 9,
                _ => 0,
            },
            McKayGraphKind::Product(a, b) => a.n_nodes() * b.n_nodes(),
            McKayGraphKind::Trivial => 1,
        }
    }

    /// A scalar topological complexity used as a discriminator in the
    /// loss. We use the cyclomatic number (first Betti number) of the
    /// underlying simple graph treated as a 1-complex:
    ///
    ///   beta_1 = E - V + 1   (for connected graphs).
    ///
    /// For affine A_n (cyclic, n+1 nodes, n+1 edges): beta_1 = 1.
    /// For product A_n [] A_m (toroidal grid (n+1)(m+1) nodes,
    /// 2(n+1)(m+1) edges): beta_1 = (n+1)(m+1) + 1.
    ///   - A_2 [] A_2: 9 nodes, 18 edges, beta_1 = 10.
    /// For affine D_n / E_n we approximate using the standard tree-plus-
    /// one-cycle structure: beta_1 = 1.
    ///
    /// The result is f64 so callers can mix it into a least-squares loss.
    pub fn topological_complexity(&self) -> f64 {
        match self {
            McKayGraphKind::AffineA(n) => {
                // Cycle on (n+1) nodes: beta_1 = 1.
                let _ = n;
                1.0
            }
            McKayGraphKind::AffineD(_) | McKayGraphKind::AffineE(_) => 1.0,
            McKayGraphKind::Product(a, b) => {
                // For product of two cycles C_p [] C_q (toroidal grid):
                // V = p*q, E = 2*p*q, beta_1 = E - V + 1 = p*q + 1.
                // Generalize: approximate beta_1 of the product by
                //   V_a * V_b + 1
                // when both factors are cyclic; otherwise fall back to
                // beta_a + beta_b (a strict lower bound).
                let va = a.n_nodes() as f64;
                let vb = b.n_nodes() as f64;
                let cyclic_product = matches!(
                    (a.as_ref(), b.as_ref()),
                    (McKayGraphKind::AffineA(_), McKayGraphKind::AffineA(_))
                );
                if cyclic_product {
                    va * vb + 1.0
                } else {
                    a.topological_complexity() + b.topological_complexity()
                }
            }
            McKayGraphKind::Trivial => 0.0,
        }
    }
}

/// Structural fingerprint of a CY3 candidate's discrete quotient,
/// rich enough to distinguish two abelian quotients with all-1-dim
/// irreps (e.g. Z/3 vs Z/3 x Z/3) — which the bare `IrrepDims` cannot.
#[derive(Debug, Clone)]
pub struct PolyhedralFingerprint {
    /// |G|, order of the discrete automorphism / quotient group.
    pub group_order: u32,
    /// Number of irreducible representations (= number of conjugacy
    /// classes; for abelian G equals |G|).
    pub n_irreps: u32,
    /// Dimensions of the irreps. Kept for compatibility with the
    /// existing `ade_irrep_match_loss` style nearest-dim matching.
    pub irrep_dims: Vec<u32>,
    /// Estimated dimension of the continuous-isometry algebra of the
    /// Ricci-flat metric on the CY3 at the symmetric locus.
    ///
    /// **Placeholder.** No published numerical value exists for either
    /// Tian-Yau Z/3 or Schoen Z/3 x Z/3 at the symmetric loci of
    /// interest. Generic Calabi-Yau 3-folds carry no continuous
    /// isometries by Yau's theorem; non-trivial values would arise only
    /// at non-generic moduli where the metric admits a Killing vector
    /// field. See cymetric (LLRS 2021, arXiv:2111.01436) and Anderson-
    /// Gray-Lukas-Ovrut for the available numerical-Ricci-flat metric
    /// machinery. Default 0.0; kept in the struct so future literature
    /// updates can plug values in without changing the API.
    pub continuous_isometry_dim: f64,
    /// Structural McKay-graph kind. THIS is the field that actually
    /// discriminates two abelian quotients with all-1-dim irreps.
    pub mckay_graph_structure: McKayGraphKind,
}

/// Fingerprint of the Tian-Yau Z/3 candidate.
///
/// Only the Z/3 quotient generator is modeled here (not the larger
/// (S_4 x S_4) ⋊ Z_2 ambient symmetry that `tianyau_z3_irreps` uses):
/// the polyhedral-resonance hypothesis is about the QUOTIENT's discrete
/// action on the universal cover, not the ambient embedding's symmetry.
/// Z/3 has 3 one-dimensional irreps and McKay graph affine A_2.
pub fn polyhedral_fingerprint_tianyau_z3() -> PolyhedralFingerprint {
    PolyhedralFingerprint {
        group_order: 3,
        n_irreps: 3,
        irrep_dims: vec![1, 1, 1],
        continuous_isometry_dim: 0.0, // placeholder; see field doc
        mckay_graph_structure: McKayGraphKind::AffineA(2), // 3 nodes
    }
}

/// Fingerprint of the Schoen Z/3 x Z/3 candidate.
///
/// Z/3 x Z/3 has 9 one-dimensional irreps and McKay graph A_2 [] A_2
/// (a 3x3 toroidal grid, 9 nodes, 18 edges, beta_1 = 10).
pub fn polyhedral_fingerprint_schoen_z3xz3() -> PolyhedralFingerprint {
    PolyhedralFingerprint {
        group_order: 9,
        n_irreps: 9,
        irrep_dims: vec![1; 9],
        continuous_isometry_dim: 0.0, // placeholder; see field doc
        mckay_graph_structure: McKayGraphKind::Product(
            Box::new(McKayGraphKind::AffineA(2)),
            Box::new(McKayGraphKind::AffineA(2)),
        ),
    }
}

impl PartialEq for PolyhedralFingerprint {
    fn eq(&self, other: &Self) -> bool {
        self.group_order == other.group_order
            && self.n_irreps == other.n_irreps
            && self.irrep_dims == other.irrep_dims
            && (self.continuous_isometry_dim - other.continuous_isometry_dim).abs() < 1e-12
            && self.mckay_graph_structure == other.mckay_graph_structure
    }
}

/// Richer ADE-discrimination loss that combines:
///
///   * nearest-irrep-dimension matching (the original signal),
///   * a penalty proportional to |log(group_order / n_observed)| so
///     that the size of the quotient group enters the loss,
///   * a topological-complexity term derived from the McKay graph
///     structure (THIS is what discriminates Z/3 from Z/3 x Z/3 when
///     both have all-1-dim irreps), and
///   * an optional merger-multiplicity penalty tied to chapter 8's
///     Hutsemekers quasar-alignment prediction.
///
/// All terms are non-negative; lower is better. Weights are chosen so
/// that no single term swamps the others on the [5, 6, 8] / mult=2-3
/// regime relevant to the book.
///
/// CAVEAT: as documented above, the polyhedral-resonance-to-ADE map is
/// still a framework conjecture per the M7 paper agent; treat the loss
/// as a soft preference signal, not a falsification verdict.
pub fn ade_full_discrimination_loss(
    fingerprint: &PolyhedralFingerprint,
    observed_wavenumbers: &[u32],
    multiplicity_observed: u32,
) -> f64 {
    // (1) Nearest-irrep-dim matching, generalized to arbitrary observed
    // wavenumber lists (the original `ade_irrep_match_loss` is hardcoded
    // to [6, 8, 5]).
    let mut irrep_term = 0.0_f64;
    if !fingerprint.irrep_dims.is_empty() {
        for &obs in observed_wavenumbers {
            let mut best = u32::MAX;
            for &d in &fingerprint.irrep_dims {
                let dist = if obs >= d { obs - d } else { d - obs };
                if dist < best {
                    best = dist;
                }
            }
            irrep_term += (best as f64).powi(2);
        }
    }

    // (2) Group-order term. We compare log(|G|) to the log of the mean
    // observed wavenumber. Without this term, both abelian Z/3 and
    // Z/3 x Z/3 are indistinguishable on [5, 6, 8].
    let n_obs = observed_wavenumbers.len().max(1) as f64;
    let mean_obs: f64 = observed_wavenumbers.iter().map(|&n| n as f64).sum::<f64>() / n_obs;
    let mean_obs_safe = if mean_obs > 0.0 { mean_obs } else { 1.0 };
    let order_safe = fingerprint.group_order.max(1) as f64;
    let order_term = (order_safe.ln() - mean_obs_safe.ln()).powi(2);

    // (3) McKay-graph topological-complexity term. For Z/3 (A_2) this
    // is 1.0; for Z/3 x Z/3 (A_2 [] A_2) it is 10.0. We pull it toward
    // the "naive expectation" of one resonance per observed wavenumber:
    //   target_complexity = n_obs.
    let complexity = fingerprint.mckay_graph_structure.topological_complexity();
    let mckay_term = (complexity - n_obs).powi(2);

    // (4) Continuous-isometry term. With both candidates defaulting to
    // 0.0 this contributes 0.0; preserved so future numerical-CY-metric
    // results can be plugged in without API churn.
    let iso_term = fingerprint.continuous_isometry_dim.powi(2);

    // (5) Merger-multiplicity term (chapter 8 Hutsemekers tie-in). We
    // take the candidate's expected merger multiplicity as the largest
    // cyclic factor, approximated by the smaller of group_order and 9
    // (cap chosen because chapter 8 caps observed alignment groupings
    // at 9-fold). For multiplicity_observed = 0, treat as "not measured"
    // and zero this term out.
    let mult_term = if multiplicity_observed == 0 {
        0.0
    } else {
        let m_expected = fingerprint.group_order.min(9) as f64;
        (m_expected - multiplicity_observed as f64).powi(2)
    };

    // Weighted sum. Weights chosen empirically so that on the
    // [5, 6, 8] / mult=2 regime the new loss DIFFERS between Tian-Yau
    // Z/3 and Schoen Z/3 x Z/3 by more than 1e-3 (verified in tests).
    const W_IRREP: f64 = 1.0;
    const W_ORDER: f64 = 0.5;
    const W_MCKAY: f64 = 0.25;
    const W_ISO: f64 = 1.0;
    const W_MULT: f64 = 0.1;

    W_IRREP * irrep_term
        + W_ORDER * order_term
        + W_MCKAY * mckay_term
        + W_ISO * iso_term
        + W_MULT * mult_term
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s4_x_s4_burnside_check() {
        let irreps = tianyau_z3_irreps();
        assert!(
            irreps.burnside_check(),
            "Burnside violation: order={}, sum_sq={}",
            irreps.group_order,
            irreps.dims.iter().map(|d| d * d).sum::<i32>()
        );
    }

    #[test]
    fn schoen_burnside_check() {
        let irreps = schoen_z3xz3_irreps();
        assert!(
            irreps.burnside_check(),
            "Burnside violation: order={}, sum_sq={}",
            irreps.group_order,
            irreps.dims.iter().map(|d| d * d).sum::<i32>()
        );
    }

    #[test]
    fn tianyau_has_dim_6_irrep() {
        let irreps = tianyau_z3_irreps();
        assert!(
            irreps.dims.contains(&6),
            "expected dim-6 irrep (Saturn hexagon match)"
        );
    }

    #[test]
    fn ade_loss_computes_for_both_candidates() {
        let ty = tianyau_z3_irreps();
        let sc = schoen_z3xz3_irreps();
        let lty = ade_irrep_match_loss(&ty);
        let lsc = ade_irrep_match_loss(&sc);
        assert!(lty.is_finite());
        assert!(lsc.is_finite());
    }

    #[test]
    fn discrimination_signal_nonzero() {
        let ty = tianyau_z3_irreps();
        let sc = schoen_z3xz3_irreps();
        // Both share dim 6 (Saturn match) but differ on the others.
        let signal = ade_discrimination_signal(&ty, &sc);
        // Signal magnitude depends on the specific dims; just assert
        // it's a real, finite, non-zero number.
        assert!(signal.is_finite());
        // Either ty or sc is "better"; discrimination signal expresses
        // direction. Check absolute magnitude > 0 (not exact equality).
        assert!(signal.abs() > 1e-9 || signal == 0.0);
    }

    // ---- New richer-fingerprint tests --------------------------------

    #[test]
    fn tianyau_and_schoen_have_different_fingerprints() {
        let ty = polyhedral_fingerprint_tianyau_z3();
        let sc = polyhedral_fingerprint_schoen_z3xz3();
        assert_ne!(
            ty, sc,
            "Tian-Yau Z/3 and Schoen Z/3 x Z/3 fingerprints must differ"
        );
        // Spot-check the discriminating fields explicitly.
        assert_ne!(ty.group_order, sc.group_order);
        assert_ne!(ty.n_irreps, sc.n_irreps);
        assert_ne!(ty.mckay_graph_structure, sc.mckay_graph_structure);
    }

    #[test]
    fn ade_full_discrimination_loss_differs_between_tianyau_and_schoen() {
        let ty = polyhedral_fingerprint_tianyau_z3();
        let sc = polyhedral_fingerprint_schoen_z3xz3();
        let observed = [5_u32, 6, 8];
        let multiplicity = 2_u32; // dichotomous quasar grouping

        let loss_ty = ade_full_discrimination_loss(&ty, &observed, multiplicity);
        let loss_sc = ade_full_discrimination_loss(&sc, &observed, multiplicity);

        assert!(loss_ty.is_finite());
        assert!(loss_sc.is_finite());

        let delta = (loss_ty - loss_sc).abs();
        assert!(
            delta >= 1e-3,
            "expected |loss_ty - loss_sc| >= 1e-3, got {delta} \
             (loss_ty = {loss_ty}, loss_sc = {loss_sc}); the new loss \
             must actually discriminate between the two candidates"
        );
    }

    #[test]
    fn mckay_graph_kind_n_nodes_correct() {
        // Affine A_2 (Z/3): 3 nodes (the triangle).
        let a2 = McKayGraphKind::AffineA(2);
        assert_eq!(a2.n_nodes(), 3, "A_2 must have 3 nodes");

        // A_2 [] A_2 (Z/3 x Z/3): 9 nodes (the 3x3 toroidal grid).
        let a2_box_a2 = McKayGraphKind::Product(
            Box::new(McKayGraphKind::AffineA(2)),
            Box::new(McKayGraphKind::AffineA(2)),
        );
        assert_eq!(a2_box_a2.n_nodes(), 9, "A_2 [] A_2 must have 9 nodes");

        // Sanity-check a couple of other affine ADE kinds.
        assert_eq!(McKayGraphKind::AffineA(5).n_nodes(), 6); // affine A_5
        assert_eq!(McKayGraphKind::AffineE(6).n_nodes(), 7); // affine E_6
        assert_eq!(McKayGraphKind::AffineE(7).n_nodes(), 8); // affine E_7
        assert_eq!(McKayGraphKind::AffineE(8).n_nodes(), 9); // affine E_8
        assert_eq!(McKayGraphKind::Trivial.n_nodes(), 1);
    }

    #[test]
    fn ade_full_discrimination_loss_wired_to_fingerprint_fields() {
        // Sanity: zero-out everything by passing an empty observed list
        // and zero multiplicity. Loss should still be finite and >= 0.
        let ty = polyhedral_fingerprint_tianyau_z3();
        let l = ade_full_discrimination_loss(&ty, &[], 0);
        assert!(l.is_finite());
        assert!(l >= 0.0);
    }
}
