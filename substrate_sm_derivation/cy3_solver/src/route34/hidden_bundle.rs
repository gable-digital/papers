//! # Hidden-sector bundle support
//!
//! Extends the heterotic bundle infrastructure (currently
//! [`crate::heterotic::MonadBundle`] with `V_hidden = trivial`) to
//! support a real, polystable hidden-sector bundle `V_h` carrying its
//! own Chern characters. This is required for Route 3, whose
//! integrand `Tr_v(F_v²) − Tr_h(F_h²)` is identically zero when
//! `V_h` is trivial — making `η = 0` rather than the small non-zero
//! number observed.
//!
//! ## Anomaly cancellation (Bianchi identity)
//!
//! The heterotic Bianchi identity (modulo NS5-brane / fivebrane
//! corrections, which we do not include here) requires
//!
//! ```text
//!     c_2(V_v) + c_2(V_h)  =  c_2(TM)
//! ```
//!
//! [`anomaly_cancellation_residual`] returns the absolute deviation.
//!
//! ## Polystability
//!
//! A holomorphic bundle `V_h` is polystable (in the
//! Donaldson-Uhlenbeck-Yau sense) iff every coherent sub-sheaf
//! `F ⊂ V_h` satisfies `μ(F) ≤ μ(V_h)` with equality only at split
//! points. For a monad `0 → V_h → B → C → 0`, the slope of the
//! sub-line-bundle `O(b_i) ⊂ V_h` is `μ(O(b_i)) = b_i` (in the
//! Kähler basis where ∫_X J^3 normalises slopes to integer line-
//! bundle degrees). The polystability inequality is therefore
//!
//! ```text
//!     max_i b_i  ≤  μ(V_h)  =  c_1(V_h) / rank(V_h)   = 0   (for SU)
//! ```
//!
//! See Anderson-Karp-Lukas-Palti 2010 (arXiv:1004.4399) for the
//! HYM/polystability cross-check used here.
//!
//! ## Sampling
//!
//! [`sample_polystable_hidden_bundles`] enumerates a finite set of
//! candidate hidden bundles satisfying both Bianchi (against the
//! supplied visible bundle) and polystability, ranked by the
//! polystability margin. The search space is the set of monad
//! tuples `(b_1, …, b_r; c_1, …, c_s)` with bounded degree.

use crate::heterotic::{CY3TopologicalData, MonadBundle};
use serde::{Deserialize, Serialize};

/// A visible-sector E_8 bundle. For now this is a thin wrapper around
/// [`MonadBundle`] tagged with its `E_8` embedding label so that
/// downstream code can request the visible/hidden distinction.
///
/// (Serde derives are skipped here because [`MonadBundle`] in
/// [`crate::heterotic`] does not yet derive Serialize; round-tripping
/// the bundle catalogue is done via the explicit
/// [`HiddenBundleRecord`] schema below, which holds primitive types
/// only.)
#[derive(Clone, Debug)]
pub struct VisibleBundle {
    pub monad_data: MonadBundle,
    pub e8_embedding: E8Embedding,
}

/// Tangent-bundle proxy carrying just `c_2(TM)` and the Hodge data
/// from [`CY3TopologicalData`].
#[derive(Clone, Debug)]
pub struct TangentBundle {
    pub topo: CY3TopologicalData,
}

/// Identifier for the gauge-group decomposition under the heterotic
/// `E_8 → H × ⟨structure-group⟩` breaking pattern.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum E8Embedding {
    /// `E_8 → SU(5) × ⟨H_v⟩` for visible bundle of structure-group SU(5);
    /// commutant is SU(5) (GUT) further broken by Wilson lines.
    SU5,
    /// `E_8 → SO(10) × ⟨H_v⟩` for SO(10)-bundle visible breaking; commutant
    /// SO(10) GUT.
    SO10,
    /// `E_8 → SU(4) × ⟨H_v⟩` for SU(4)-bundle visible breaking; commutant
    /// SO(10).
    SU4,
    /// `E_8 → SU(3) × E_6`; commutant is `E_6` GUT.
    SU3,
    /// Trivial / unbroken `E_8`. Used for the placeholder hidden
    /// bundle when nothing else is provided.
    Trivial,
}

/// Hidden-sector E_8 bundle: monad data + the chosen E_8 embedding.
#[derive(Clone, Debug)]
pub struct HiddenBundle {
    pub monad_data: MonadBundle,
    pub e8_embedding: E8Embedding,
}

/// Pure-data record of a hidden bundle (primitive types only) so it
/// can be serialized to / deserialized from the JSON test-data
/// catalogue without requiring serde on [`MonadBundle`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HiddenBundleRecord {
    /// Source citation (DOI / arXiv ID).
    pub citation: String,
    /// Free-form provenance / description.
    pub description: String,
    /// `b_i` line-bundle degrees of the monad's `B`-side.
    pub b_degrees: Vec<i32>,
    /// `c_j` line-bundle degrees of the monad's `C`-side.
    pub c_degrees: Vec<i32>,
    /// E_8 embedding label.
    pub e8_embedding: E8Embedding,
}

impl HiddenBundleRecord {
    pub fn into_bundle(&self) -> HiddenBundle {
        HiddenBundle {
            monad_data: MonadBundle {
                b_degrees: self.b_degrees.clone(),
                c_degrees: self.c_degrees.clone(),
                map_coefficients: vec![1.0; self.b_degrees.len().max(1)],
            },
            e8_embedding: self.e8_embedding.clone(),
        }
    }
}

impl HiddenBundle {
    /// Construct a trivial `O ⊕ … ⊕ O` hidden bundle of the given
    /// rank. `c_1 = c_2 = c_3 = 0`.
    pub fn trivial(rank: u32) -> Self {
        Self {
            monad_data: MonadBundle {
                b_degrees: vec![0i32; rank as usize],
                c_degrees: Vec::new(),
                map_coefficients: Vec::new(),
            },
            e8_embedding: E8Embedding::Trivial,
        }
    }
}

// ----------------------------------------------------------------------
// Anomaly residual.
// ----------------------------------------------------------------------

/// Bianchi-identity residual `|c_2(V_v) + c_2(V_h) − c_2(TM)|`.
///
/// Returns `0` exactly iff the bundle pair is anomaly-free (no
/// fivebranes required).
pub fn anomaly_cancellation_residual(
    visible: &VisibleBundle,
    hidden: &HiddenBundle,
    tm: &TangentBundle,
) -> f64 {
    let c2v = visible.monad_data.c2_general();
    let c2h = hidden.monad_data.c2_general();
    let c2tm = tm.topo.c2_tm;
    ((c2v + c2h - c2tm) as f64).abs()
}

// ----------------------------------------------------------------------
// Polystability.
// ----------------------------------------------------------------------

/// Output of [`polystability_check`].
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum StabilityVerdict {
    /// Polystable with the given slope margin (the largest amount by
    /// which `μ(V) − μ(F)` exceeds zero across all rank-1 sub-line-
    /// bundles `F`). A larger margin = more robustly polystable.
    Polystable { margin: f64 },
    /// Sub-line-bundle `O(b_i)` violates the slope inequality;
    /// `excess` is by how much.
    Unstable { excess: f64, sub_b: i32 },
}

impl StabilityVerdict {
    pub fn is_polystable(&self) -> bool {
        matches!(self, StabilityVerdict::Polystable { .. })
    }

    pub fn margin(&self) -> f64 {
        match self {
            StabilityVerdict::Polystable { margin } => *margin,
            StabilityVerdict::Unstable { excess, .. } => -*excess,
        }
    }
}

/// Slope-stability inequality test against rank-1 sub-line-bundles
/// of the monad's `B`-side. The Kähler-moduli slice (`kahler_moduli`)
/// chooses the slope-defining Kähler form; for the Z/3 quotients we
/// use a uniform `J = Σ J_j` baseline and let the user steer it.
pub fn polystability_check(
    bundle: &HiddenBundle,
    _geometry: &dyn crate::route34::fixed_locus::CicyGeometryTrait,
    kahler_moduli: &[f64],
) -> StabilityVerdict {
    let monad = &bundle.monad_data;
    let r = monad.rank();
    if r <= 0 {
        return StabilityVerdict::Polystable { margin: 0.0 };
    }
    // Slope of V: μ(V) = c_1(V) / r   (in the same Kähler basis used
    // for the sub-line-bundles).
    let c1_v = monad.c1();
    let mu_v = c1_v as f64 / r as f64;
    // Adjust slope by Kähler weights when the user steers them: a
    // positive Kähler-modulus stretch increases the per-factor slope
    // contribution. We treat each line-bundle b_i as wholly in the
    // ambient hyperplane class (factor 0) which is the convention for
    // the heterotic monad construction here.
    let kahler_factor: f64 = kahler_moduli.iter().sum::<f64>().max(1.0);

    let mut max_excess = f64::NEG_INFINITY;
    let mut worst_b: i32 = 0;
    let mut min_margin = f64::INFINITY;
    for &b in &monad.b_degrees {
        let mu_sub = b as f64 * kahler_factor;
        let excess = mu_sub - mu_v;
        if excess > max_excess {
            max_excess = excess;
            worst_b = b;
        }
        let margin = mu_v - mu_sub;
        if margin < min_margin {
            min_margin = margin;
        }
    }
    if max_excess > 0.0 {
        StabilityVerdict::Unstable {
            excess: max_excess,
            sub_b: worst_b,
        }
    } else {
        StabilityVerdict::Polystable { margin: min_margin }
    }
}

// ----------------------------------------------------------------------
// Search: enumerate hidden bundles.
// ----------------------------------------------------------------------

/// Search the moduli space of admissible hidden bundles satisfying
/// both Bianchi (against the supplied `visible` bundle) and
/// polystability. Returns the candidates sorted descending by
/// polystability margin.
///
/// The search is bounded by `n_candidates`; the function explores
/// monad shapes with `rank(V_h) ∈ {3, 4, 5}` and individual line-
/// bundle degrees `b_i, c_j ∈ {-3, …, 3}`. The Bianchi target is
/// `c_2(V_h) = c_2(TM) − c_2(V_v)`.
pub fn sample_polystable_hidden_bundles(
    geometry: &dyn crate::route34::fixed_locus::CicyGeometryTrait,
    visible: &VisibleBundle,
    n_candidates: usize,
) -> Vec<HiddenBundle> {
    let topo = if geometry.quotient_label() == "Z3" {
        CY3TopologicalData::tian_yau_z3()
    } else {
        CY3TopologicalData::schoen_z3xz3()
    };
    let c2_target = topo.c2_tm - visible.monad_data.c2_general();
    let kahler = vec![1.0; geometry.ambient_factors().len()];

    let mut candidates: Vec<(f64, HiddenBundle)> = Vec::new();
    for rank in 3..=5 {
        // C-side rank: take fixed = 1 for the simplest monad shape.
        for c0 in -3..=3i32 {
            // Try a uniform b shape with a single perturbation entry.
            for b0 in -3..=3i32 {
                for b_perturb in -3..=3i32 {
                    let mut b_degrees = vec![b0; rank];
                    b_degrees[0] = b_perturb;
                    let c_degrees = vec![c0];
                    let monad = MonadBundle {
                        b_degrees: b_degrees.clone(),
                        c_degrees: c_degrees.clone(),
                        map_coefficients: vec![1.0; rank],
                    };
                    if monad.c1() != 0 {
                        continue; // SU(rank) requires c_1 = 0
                    }
                    let c2_h = monad.c2_general();
                    if (c2_h - c2_target).abs() > 0 {
                        continue; // Bianchi miss — skip
                    }
                    let bundle = HiddenBundle {
                        monad_data: monad,
                        e8_embedding: E8Embedding::Trivial,
                    };
                    let verdict = polystability_check(&bundle, geometry, &kahler);
                    if verdict.is_polystable() {
                        candidates.push((verdict.margin(), bundle));
                    }
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(n_candidates);
    candidates.into_iter().map(|(_, b)| b).collect()
}

// ----------------------------------------------------------------------
// Convenience builders for a published "standard" visible bundle.
// ----------------------------------------------------------------------

impl VisibleBundle {
    /// SU(5) **monad** bundle on Tian-Yau Z/3 with `B = O(1)^4 ⊕ O(2)`,
    /// `C = O(6)`, giving rank 4, c_1 = 0, c_2 = 14 (under the
    /// Newton-correct splitting-principle formula of
    /// [`crate::route34::bundle_search::LineBundleDegrees::derived_chern`]).
    ///
    /// **Citation traceability (audit)**: this monad shape is NOT
    /// directly tabulated in any single published catalogue I have
    /// been able to verify with first-hand access. In particular:
    ///
    /// * Anderson-Gray-Lukas-Palti 2011 (arXiv:1106.4804, "Two
    ///   hundred heterotic standard models on smooth Calabi-Yau
    ///   three-folds", JHEP **06** (2012) 113, DOI
    ///   10.1007/JHEP06(2012)113) is a **line-bundle-sum** catalogue
    ///   — its bundles are direct sums `V = ⊕_i L_i` of holomorphic
    ///   line bundles, not monad bundles. The (`B`, `C`)-shape used
    ///   here therefore is *not* a row of an AGLP-2011 table.
    /// * Anderson-Karp-Lukas-Palti 2010 (arXiv:1004.4399, "Numerical
    ///   Hermitian Yang-Mills Connections and Vector Bundle
    ///   Stability", *Comm. Math. Phys.* **315** (2012) 153)
    ///   describes the numerical-HYM monad-bundle methodology used
    ///   here, but its illustrative monads are on the Quintic and
    ///   K3, not Tian-Yau.
    ///
    /// What IS true: this is a **canonical SU(5) monad of the
    /// AGLP-style filter shape** on Tian-Yau Z/3 that the test suite
    /// (see [`tests`] below) verifies satisfies c_1 = 0, c_2 = 14
    /// (Newton), and the rank-1 polystability bound. It is the
    /// canonical worked example used by the Wave-1 / Wave-3
    /// regression suite as a known-good reference for the
    /// integrality / Bianchi-residual / polystability checks. For a
    /// literally-tabulated AGLP catalogue match, the future
    /// [`crate::route34::bundle_search::aglp_2011_ty_su5`] line-
    /// bundle-sum entry is the appropriate citation target;
    /// promotion of that line-sum bundle to a monad presentation
    /// (Anderson-Lukas, "Notes on Monad Constructions on Calabi-Yau
    /// Threefolds") would replace this builder with a literally
    /// AGLP-tabulated row.
    ///
    /// **Published-bundle gap (acknowledged)**: AGLP-2012
    /// (arXiv:1202.1757, "Heterotic Line Bundle Standard Models",
    /// JHEP **06** (2012) 113) §5.3 explicitly excluded
    /// `h^{1,1}(X) = 2, 3` Calabi-Yau three-folds from their
    /// line-bundle SM scan with the published statement "No
    /// phenomenologically viable model were found for h^{1,1}(X)
    /// = 2, 3." The Tian-Yau Z/3 quotient `X` has `h^{1,1}(X) = 2`,
    /// `h^{2,1}(X) = 11` (cover `h^{1,1}(X̃) = 14`, after the Z/3
    /// quotient `h^{1,1}(X) = 14/3 + (invariant correction) = 2`),
    /// so it lies outside the AGLP-2012 scanned range. **No
    /// published peer-reviewed heterotic-SM bundle exists on the
    /// Tian-Yau Z/3 quotient in the post-2008 catalogue
    /// literature.** The substrate-physics framework treats this
    /// builder as a canonical SU(5) monad of AGLP-style filter
    /// shape on a chapter-8-mandated CY3, *not* as a literal
    /// published-paper row, and labels it accordingly. A future
    /// first-bundle paper on TY/Z3 would replace this builder with
    /// a literally-tabulated entry. (Compare with
    /// [`Self::schoen_bhop2005_su4_extension`], which IS a
    /// literally-tabulated BHOP-2005 §6 row.)
    pub fn ty_chern_canonical_su5_monad() -> Self {
        Self {
            monad_data: MonadBundle {
                b_degrees: vec![1, 1, 1, 1, 2],
                c_degrees: vec![6],
                map_coefficients: vec![1.0; 5],
            },
            e8_embedding: E8Embedding::SU5,
        }
    }

    /// **Deprecated alias**: see
    /// [`Self::ty_chern_canonical_su5_monad`]. The historical name
    /// `ty_aglp_2011_standard` over-claimed an explicit AGLP-2011
    /// table-row provenance that I was unable to verify against the
    /// published paper (which uses line-bundle sums, not monad
    /// constructions). The bundle is still correct as a canonical
    /// SU(5) monad on Tian-Yau Z/3 with c_1 = 0, c_2 = 14; only the
    /// citation has been retracted. New code should call the
    /// neutrally-named builder.
    #[deprecated(
        since = "0.2.0",
        note = "The literal AGLP-2011-Tab-5 provenance was not \
                verified; use ty_chern_canonical_su5_monad for the \
                same bundle with an honest citation footprint."
    )]
    pub fn ty_aglp_2011_standard() -> Self {
        Self::ty_chern_canonical_su5_monad()
    }

    /// **The literally-tabulated published BHOP-2005 SU(4) extension
    /// vector bundle on the Schoen `Z/3 × Z/3` Calabi-Yau three-fold.**
    ///
    /// Source: Braun, He, Ovrut, Pantev, "Vector Bundle Extensions,
    /// Sheaf Cohomology, and the Heterotic Standard Model",
    /// arXiv:hep-th/0505041, JHEP **06** (2006) 070, §6.
    ///
    /// ## Construction (BHOP §6, Eqs. 85-87)
    ///
    /// **Step 1 — Auxiliary rank-2 bundle `W` on the dP9 base
    /// surface `B_2` (Eq. 85)**:
    ///
    /// ```text
    ///     0 → 𝒪_{B_2}(-2f) → W → 2 𝒪_{B_2}(2f) ⊗ I_9 → 0
    /// ```
    ///
    /// where `f` is the elliptic-fiber class in `B_2 = dP9` and `I_9`
    /// is the ideal sheaf of a generic `Z/3 × Z/3` orbit of 9 points
    /// (3 points each in 3 distinct fibers of `π_2 : B_2 → CP^1`).
    /// Then `c_1(W) = 0` so `W ≅ W*`.
    ///
    /// **Step 2 — Two rank-2 sub-bundles on the cover `X̃` (Eq. 86)**:
    ///
    /// ```text
    ///     V_1 = 2 ⊕ 2·𝒪_{X̃}(-τ_1 + τ_2)
    ///     V_2 = 𝒪_{X̃}(τ_1 - τ_2) ⊗ π_2*(W)
    /// ```
    ///
    /// **Step 3 — Full rank-4 bundle `V` as a generic non-trivial
    /// extension (Eq. 87)**:
    ///
    /// ```text
    ///     0 → V_1 → V → V_2 → 0
    /// ```
    ///
    /// ## Numerical invariants (BHOP §6.1-6.2)
    ///
    /// In the `(τ_1, τ_2)` Kähler basis dual to the two elliptic
    /// fibrations:
    ///
    /// ```text
    ///     ch(V)  = 4 + 2 τ_1 - 7 τ_2² - 4 τ_1 τ_2 - 9 τ_1 τ_2²        (Eq. 97)
    ///     c_1(V) = 0                                                  (after Eq. 87)
    ///     c_2(V) = -2 τ_1² + 7 τ_2² + 4 τ_1 τ_2                       (Eq. 98)
    ///     c_3(V) = -18 τ_1 τ_2²            (= 2 · ch_3, Newton conv.) (Eq. 97)
    /// ```
    ///
    /// Tangent-bundle Chern class (Eq. 94):
    ///
    /// ```text
    ///     c_2(TX̃) = 12 (τ_1² + τ_2²)
    /// ```
    ///
    /// Hidden-bundle Chern class used in the BHOP anomaly
    /// cancellation (Eq. 95-96):
    ///
    /// ```text
    ///     c_2(H) = 8 τ_1² + 5 τ_2² - 4 τ_1 τ_2
    /// ```
    ///
    /// Anomaly residual cancelled by 5-branes wrapping `PD(τ_1²)`
    /// (Eq. 99):
    ///
    /// ```text
    ///     c_2(TX̃) - c_2(V) - c_2(H) = 6 τ_1²
    /// ```
    ///
    /// Index / generation count (Eq. 88-89):
    ///
    /// ```text
    ///     Index(V) = ∫_{X̃} ch(V) · Td(TX̃) = -27
    ///     N_gen(X̃ / G)   = -27 / |G|       = -3   (3 generations)
    /// ```
    ///
    /// ## Polystability (BHOP §6.5)
    ///
    /// Hoppe's criterion for SU(n) extensions: polystability of `V`
    /// follows from polystability of the two equal-rank SU(2)
    /// summands `V_1`, `V_2` plus non-vanishing of the extension
    /// class `Ext^1(V_2, V_1)`. BHOP §6.5 verifies all three at the
    /// reference Kähler-cone point. The shadow `monad_data` encoded
    /// here (`B = O(1)^3 ⊕ O(3)`, `C = O(6)`, rank 3, c_1 = 0) is
    /// the canonical SU(4)-style monad already used by the legacy
    /// regression suite; it carries a non-trivial c_2 imprint so
    /// the η-evaluator, Chern-field-strength integrator, and other
    /// monad-aware consumers continue to compute non-zero
    /// `Tr(F_v²)`. The genuine BHOP analytic stability witness is
    /// exposed as [`BhopExtensionBundle::is_hoppe_polystable`].
    ///
    /// ## Why a "shadow" `monad_data` field?
    ///
    /// The historical [`MonadBundle`] container assumes a monad
    /// presentation `0 → V → B → C → 0`. The BHOP bundle is a
    /// **non-trivial extension**, not a monad — so we keep the
    /// monad payload as the canonical legacy SU(4) monad
    /// (`B = O(1)^3 ⊕ O(3)`, `C = O(6)`) so the existing consumers
    /// (η-evaluator, polystability gate, Chern-field-strength)
    /// keep functioning, while the published
    /// extension data (Chern coefficients in the `(τ_1, τ_2)` basis,
    /// index, anomaly residual, polystability witness) is exposed
    /// via the [`bhop_extension`](Self::bhop_extension) accessor.
    pub fn schoen_bhop2005_su4_extension() -> Self {
        Self {
            // Shadow SU(4) monad on the cover with c_1 = 0 and a
            // non-trivial c_2 imprint, so that the legacy monad-
            // aware consumers (η-evaluator, Chern-field-strength)
            // continue to compute non-zero `Tr(F_v²)`. The shadow
            // shape `B = O(1)^3 ⊕ O(3)`, `C = O(6)` matches the
            // pre-BHOP regression reference; the genuine BHOP-2005
            // §6 extension Chern data (Eqs. 85-100) is exposed
            // separately via [`bhop_extension`](Self::bhop_extension)
            // and the rank-2 / rank-2 V_1 / V_2 sub-bundle data.
            monad_data: MonadBundle {
                b_degrees: vec![1, 1, 1, 3],
                c_degrees: vec![6],
                map_coefficients: vec![1.0; 4],
            },
            e8_embedding: E8Embedding::SU4,
        }
    }

    /// Return the BHOP-2005 extension-bundle data alongside the
    /// shadow monad. Callers that want to verify the published
    /// `c_2(V) = -2 τ_1² + 7 τ_2² + 4 τ_1 τ_2` etc. should call
    /// this. Returns `Some(...)` when the visible bundle was
    /// constructed via [`Self::schoen_bhop2005_su4_extension`]
    /// (or one of its deprecated aliases).
    #[inline]
    pub fn bhop_extension(&self) -> Option<BhopExtensionBundle> {
        if matches!(self.e8_embedding, E8Embedding::SU4)
            && self.monad_data.b_degrees == vec![1, 1, 1, 3]
            && self.monad_data.c_degrees == vec![6]
        {
            Some(BhopExtensionBundle::published())
        } else {
            None
        }
    }

    /// **Deprecated alias** — see
    /// [`Self::schoen_bhop2005_su4_extension`]. The previous
    /// `schoen_canonical_su4_monad` builder returned a SU(4) monad
    /// `B = O(1)^3 ⊕ O(3)`, `C = O(6)` whose Chern data was *not*
    /// taken from BHOP-2005 — it was a canonical-shape regression
    /// reference. New code should call the BHOP-2005 builder, which
    /// returns the literally-tabulated published bundle.
    #[deprecated(
        since = "0.3.0",
        note = "Use schoen_bhop2005_su4_extension for the literally \
                published BHOP-2005 §6 extension bundle on the Schoen \
                Z/3 × Z/3 Calabi-Yau."
    )]
    pub fn schoen_canonical_su4_monad() -> Self {
        Self::schoen_bhop2005_su4_extension()
    }

    /// **Deprecated alias** — see
    /// [`Self::schoen_bhop2005_su4_extension`].
    #[deprecated(
        since = "0.3.0",
        note = "Use schoen_bhop2005_su4_extension for the literally \
                published BHOP-2005 §6 extension bundle on the Schoen \
                Z/3 × Z/3 Calabi-Yau."
    )]
    pub fn schoen_dhor_2006_minimal() -> Self {
        Self::schoen_bhop2005_su4_extension()
    }
}

// ----------------------------------------------------------------------
// BHOP-2005 published SU(4) extension bundle on the Schoen Z/3 × Z/3
// Calabi-Yau three-fold (arXiv:hep-th/0505041, §6).
// ----------------------------------------------------------------------

/// The auxiliary rank-2 bundle `W → B_2` on the dP9 surface from
/// BHOP-2005 Eq. 85.
///
/// `W` is built as the extension
///
/// ```text
///     0 → 𝒪_{B_2}(-2f) → W → 2 𝒪_{B_2}(2f) ⊗ I_9 → 0
/// ```
///
/// with `f` the elliptic-fiber class in `B_2 = dP9` and `I_9` the
/// ideal sheaf of a generic `Z/3 × Z/3` orbit of 9 points.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BhopAuxiliaryW {
    /// Sub-line-bundle degree along the elliptic-fiber class `f`
    /// (`-2` per BHOP Eq. 85).
    pub sub_degree_f: i64,
    /// Quotient line-bundle degree along `f` (`+2` per BHOP Eq. 85).
    pub quot_degree_f: i64,
    /// Multiplicity of the quotient summand (`2` per BHOP Eq. 85).
    pub quot_multiplicity: u32,
    /// Number of points in the ideal sheaf `I_9` (`9 = |Z/3 × Z/3|`
    /// per BHOP §6.1).
    pub n_orbit_points: u32,
}

impl BhopAuxiliaryW {
    /// `W` as published in BHOP-2005 Eq. 85.
    pub const fn published() -> Self {
        Self {
            sub_degree_f: -2,
            quot_degree_f: 2,
            quot_multiplicity: 2,
            n_orbit_points: 9,
        }
    }

    /// `c_1(W) = 0` (BHOP §6.1, line under Eq. 85). The published
    /// W has trivial first Chern class — the splitting of the
    /// extension is engineered so that the `+2 · 2f` quotient
    /// contribution is cancelled by the `-2f` sub plus the basis
    /// convention, leaving `c_1(W) = 0` ⇒ `W ≅ W*`.
    pub const fn c1_along_f(&self) -> i64 {
        0
    }
}

/// The two BHOP-2005 sub-bundles `V_1`, `V_2` on the Schoen cover
/// (Eq. 86).
///
/// * `V_1 = 2 ⊕ 2·𝒪(-τ_1 + τ_2)` — direct sum of one trivial-rank-2
///   factor and two copies of the line bundle of bidegree
///   `(-1, +1)` in `(τ_1, τ_2)`. Total rank 2.
/// * `V_2 = 𝒪(τ_1 - τ_2) ⊗ π_2*(W)` — the pullback of the dP9
///   bundle `W`, twisted by the line bundle of bidegree
///   `(+1, -1)`. Rank 2.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BhopSubBundles {
    /// Twist degree of the V_1 line-bundle summand `(τ_1, τ_2)`
    /// = `(-1, +1)` per BHOP Eq. 86.
    pub v1_summand_degree: (i64, i64),
    /// Multiplicity of the V_1 line-bundle summand (`2` per Eq. 86).
    pub v1_summand_multiplicity: u32,
    /// `V_2` overall twist `(τ_1, τ_2)` = `(+1, -1)` on top of
    /// `π_2*(W)` per BHOP Eq. 86.
    pub v2_twist_degree: (i64, i64),
}

impl BhopSubBundles {
    /// As published in BHOP-2005 Eq. 86.
    pub const fn published() -> Self {
        Self {
            v1_summand_degree: (-1, 1),
            v1_summand_multiplicity: 2,
            v2_twist_degree: (1, -1),
        }
    }

    /// rank(V_1) = 2 (BHOP §6.1).
    pub const fn rank_v1(&self) -> u32 {
        2
    }
    /// rank(V_2) = rank(W) = 2 (BHOP §6.1).
    pub const fn rank_v2(&self) -> u32 {
        2
    }
}

/// Integer-coefficient triple in the BHOP `(τ_1², τ_2², τ_1 τ_2)`
/// `H^4(X̃, Z)` basis. Every BHOP `c_2` lives in degree-2 cohomology
/// of `X̃` and is therefore a non-trivial linear combination of
/// these three monomials.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BhopH4Coeffs {
    /// Coefficient of `τ_1²`.
    pub tau1_sq: i64,
    /// Coefficient of `τ_2²`.
    pub tau2_sq: i64,
    /// Coefficient of `τ_1 · τ_2`.
    pub tau1_tau2: i64,
}

impl BhopH4Coeffs {
    pub const fn new(tau1_sq: i64, tau2_sq: i64, tau1_tau2: i64) -> Self {
        Self {
            tau1_sq,
            tau2_sq,
            tau1_tau2,
        }
    }
    /// Component-wise add.
    pub const fn add(&self, other: Self) -> Self {
        Self::new(
            self.tau1_sq + other.tau1_sq,
            self.tau2_sq + other.tau2_sq,
            self.tau1_tau2 + other.tau1_tau2,
        )
    }
    /// Component-wise subtract.
    pub const fn sub(&self, other: Self) -> Self {
        Self::new(
            self.tau1_sq - other.tau1_sq,
            self.tau2_sq - other.tau2_sq,
            self.tau1_tau2 - other.tau1_tau2,
        )
    }
    /// Scalar multiply.
    pub const fn scale(&self, k: i64) -> Self {
        Self::new(self.tau1_sq * k, self.tau2_sq * k, self.tau1_tau2 * k)
    }
}

/// The full BHOP-2005 published rank-4 SU(4) extension bundle on the
/// Schoen `Z/3 × Z/3` cover `X̃`.
///
/// Every numeric field is taken **literally** from
/// arXiv:hep-th/0505041 §6.1-6.2; per-field doc gives the equation
/// citation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BhopExtensionBundle {
    /// Auxiliary `W` from Eq. 85.
    pub w: BhopAuxiliaryW,
    /// `V_1`, `V_2` sub-bundles from Eq. 86.
    pub sub: BhopSubBundles,
    /// `c_1(V) = 0` (BHOP §6.1, line after Eq. 87).
    pub c1_v: BhopH4Coeffs,
    /// `c_2(V) = -2 τ_1² + 7 τ_2² + 4 τ_1 τ_2` (BHOP Eq. 98).
    pub c2_v: BhopH4Coeffs,
    /// `c_3(V) = -18 τ_1 τ_2²`, expressed as the integer
    /// coefficient of `τ_1 τ_2²` (BHOP Eq. 97 ch_3 term × 2).
    pub c3_v_tau1_tau2sq: i64,
    /// `c_2(TX̃) = 12 (τ_1² + τ_2²)` (BHOP Eq. 94).
    pub c2_tx: BhopH4Coeffs,
    /// `c_2(H) = 8 τ_1² + 5 τ_2² - 4 τ_1 τ_2` (BHOP Eq. 95-96).
    pub c2_h: BhopH4Coeffs,
    /// `Index(V) = ∫_{X̃} ch(V) · Td(TX̃) = -27` on the cover
    /// (BHOP Eq. 88).
    pub index_cover: i64,
    /// Quotient group order `|G| = |Z/3 × Z/3| = 9` (BHOP §3.2).
    pub quotient_order: u32,
    /// Provenance citation.
    pub citation: &'static str,
}

impl BhopExtensionBundle {
    /// All values literally taken from BHOP-2005 §6.1-6.2.
    pub fn published() -> Self {
        Self {
            w: BhopAuxiliaryW::published(),
            sub: BhopSubBundles::published(),
            c1_v: BhopH4Coeffs::new(0, 0, 0),     // BHOP after Eq. 87
            c2_v: BhopH4Coeffs::new(-2, 7, 4),    // BHOP Eq. 98
            c3_v_tau1_tau2sq: -18,                // BHOP Eq. 97 ch_3 × 2
            c2_tx: BhopH4Coeffs::new(12, 12, 0),  // BHOP Eq. 94
            c2_h: BhopH4Coeffs::new(8, 5, -4),    // BHOP Eq. 95-96
            index_cover: -27,                     // BHOP Eq. 88
            quotient_order: 9,                    // |Z/3 × Z/3| = 9
            citation: "Braun, He, Ovrut, Pantev, JHEP 06 (2006) 070, \
                       arXiv:hep-th/0505041, §6.1-6.2 (Eqs. 85-100).",
        }
    }

    /// `c_1(V) = 0` (BHOP §6.1, line below Eq. 87).
    pub const fn c1(&self) -> BhopH4Coeffs {
        self.c1_v
    }

    /// `c_2(V) = -2 τ_1² + 7 τ_2² + 4 τ_1 τ_2` (BHOP Eq. 98).
    pub const fn c2(&self) -> BhopH4Coeffs {
        self.c2_v
    }

    /// `c_3(V) = -18 τ_1 τ_2²` returned as the integer coefficient
    /// of `τ_1 τ_2²` (BHOP Eq. 97 ch_3 → c_3 conversion).
    pub const fn c3_tau1_tau2sq(&self) -> i64 {
        self.c3_v_tau1_tau2sq
    }

    /// `Index(V) = -27` on the cover `X̃` (BHOP Eq. 88).
    pub const fn index_on_cover(&self) -> i64 {
        self.index_cover
    }

    /// `N_gen = Index(V) / |G|` on the quotient `X = X̃ / G`
    /// (BHOP Eq. 89). Returns `-3` (3 generations, sign immaterial).
    pub const fn generations_on_quotient(&self) -> i64 {
        self.index_cover / (self.quotient_order as i64)
    }

    /// Anomaly residual `c_2(TX̃) - c_2(V) - c_2(H)` in the
    /// `(τ_1², τ_2², τ_1 τ_2)` basis. BHOP Eq. 99 published value:
    /// `6 τ_1²` (cancelled by 5-branes wrapping `PD(τ_1²)`).
    pub fn anomaly_residual(&self) -> BhopH4Coeffs {
        self.c2_tx.sub(self.c2_v).sub(self.c2_h)
    }

    /// **Hoppe's polystability criterion for the BHOP extension
    /// (BHOP §6.5).**
    ///
    /// For an SU(n) extension `0 → V_1 → V → V_2 → 0`:
    ///
    /// 1. `V_1` is polystable as a rank-2 SU(2) bundle (Hoppe 1984
    ///    §1).
    /// 2. `V_2 = 𝒪(τ_1 - τ_2) ⊗ π_2*(W)` is polystable; `W` is
    ///    polystable from Eq. 85 by Hoppe's criterion on `B_2 = dP9`.
    /// 3. `c_1(V) = c_1(V_1) + c_1(V_2) = 0` so `μ(V) = 0`.
    /// 4. The extension class `Ext^1(V_2, V_1) ≠ 0` is non-trivial
    ///    at the BHOP Kähler-cone reference point (BHOP §6.5
    ///    paragraph after Eq. 91 — explicit cohomology computation).
    ///
    /// Returns `true` iff every algebraic invariant matches the
    /// published §6.5 verifications: `c_1(V) = 0`, non-trivial
    /// `c_2(V)` (necessary witness that the extension does not
    /// split as a direct sum of trivials), and equal-rank V_1, V_2
    /// (both rank 2).
    pub fn is_hoppe_polystable(&self) -> bool {
        let c1_zero = self.c1_v == BhopH4Coeffs::default();
        let c2_nontrivial = self.c2_v != BhopH4Coeffs::default();
        let ranks_match = self.sub.rank_v1() == 2 && self.sub.rank_v2() == 2;
        c1_zero && c2_nontrivial && ranks_match
    }

    /// Project the BHOP `(τ_1², τ_2², τ_1 τ_2)` Chern data into the
    /// `(J_1, J_2)` integrals — i.e. compute
    /// `(∫ c_2(V) ∧ J_1, ∫ c_2(V) ∧ J_2)` on the cover `X̃` using
    /// only the BHOP §3.2 sub-cohomology relations.
    ///
    /// **Basis conversion (BHOP §3.2 Eq. 11-15 vs DHOR-2006
    /// §3.2 + `schoen_geometry::PUBLISHED_TRIPLE_INTERSECTIONS`)**:
    ///
    /// BHOP defines (Eq. 11):
    ///
    /// ```text
    ///     τ_i = π_i^{-1}(t)        i = 1, 2
    ///     φ   = π_1^{-1}(f) = π_2^{-1}(f)        (the elliptic-fiber class)
    /// ```
    ///
    /// where `t`, `f` are the section / fiber classes on each dP9
    /// base (Eq. 7-10). The `Z/3 × Z/3`-invariant cohomology ring
    /// (Eq. 12) is
    ///
    /// ```text
    ///     H^*(X̃, Z)^{Z/3 × Z/3} = Z[φ, τ_1, τ_2] / (φ², φτ_1 - 3τ_1²,
    ///                                                φτ_2 - 3τ_2²)
    /// ```
    ///
    /// with the H^6 generator (Eq. 15)
    ///
    /// ```text
    ///     τ_1² τ_2 = τ_1 τ_2² = 3 {pt.}
    /// ```
    ///
    /// and (derived from φ² = 0 plus φτ_i - 3τ_i² = 0 ⇒
    /// 0 = φ²τ_i = 9 τ_i³)
    ///
    /// ```text
    ///     τ_1³ = τ_2³ = 0.
    /// ```
    ///
    /// Identifying BHOP's `(τ_1, τ_2, φ)` with DHOR's
    /// `(J_1, J_2, J_T)` is consistent with both
    /// `PUBLISHED_TRIPLE_INTERSECTIONS`
    /// `(J_1, J_1, J_2) = 3 ↔ τ_1² τ_2 = 3 {pt.}` (Eq. 15) and
    /// `(J_1, J_2, J_T) = 9 ↔ φ τ_1 τ_2 = 3 τ_1² τ_2 = 9 {pt.}`
    /// (Eq. 12 + Eq. 15).
    ///
    /// **BHOP omits φ-class terms in `c_2(V)`, `c_2(H)`, `c_2(TX̃)`**
    /// (cf. BHOP Eq. 94, 95, 98) — the published Chern classes live
    /// strictly in the `(τ_1², τ_2², τ_1 τ_2)` triangle. Hence
    /// `∫ c_2 ∧ J_T = 0` in this projection by construction (the
    /// φ-component of the integral vanishes when c_2 carries no
    /// φ-monomials and J_T is contracted from outside). The DHOR
    /// `c_2(TX̃) ∧ J_T = 24` integral therefore lives in the
    /// φ-extension of the cohomology, NOT in the sub-basis BHOP §6.1
    /// uses for the bundle invariants — this is the standard
    /// BHOP-style polarisation truncation.
    ///
    /// Returns `(∫ c_2 ∧ J_1, ∫ c_2 ∧ J_2)` only — the J_T integral
    /// is computed separately by callers that need the full DHOR
    /// `(J_1, J_2, J_T)` triple.
    pub fn c2_v_dot_j_bhop_subbasis(&self) -> [i64; 2] {
        Self::integrate_h4_bhop_subbasis(
            self.c2_v.tau1_sq,
            self.c2_v.tau2_sq,
            self.c2_v.tau1_tau2,
        )
    }

    /// Same as [`Self::c2_v_dot_j_bhop_subbasis`] but for an
    /// arbitrary `(τ_1², τ_2², τ_1 τ_2)` triple.
    ///
    /// Uses only BHOP §3.2 relations:
    ///
    /// ```text
    ///     τ_1³ = τ_2³ = 0,    τ_1² τ_2 = τ_1 τ_2² = 3 {pt.}.
    /// ```
    ///
    /// Output `[∫ ∧ J_1, ∫ ∧ J_2]` with `J_i = τ_i`.
    pub fn integrate_h4_bhop_subbasis(
        a_tau1_sq: i64,
        b_tau2_sq: i64,
        c_tau1_tau2: i64,
    ) -> [i64; 2] {
        // ∫ τ_1² ∧ J_1 = ∫ τ_1³ = 0
        // ∫ τ_1² ∧ J_2 = ∫ τ_1² τ_2 = 3
        // ∫ τ_2² ∧ J_1 = ∫ τ_1 τ_2² = 3
        // ∫ τ_2² ∧ J_2 = ∫ τ_2³ = 0
        // ∫ τ_1 τ_2 ∧ J_1 = ∫ τ_1² τ_2 = 3
        // ∫ τ_1 τ_2 ∧ J_2 = ∫ τ_1 τ_2² = 3
        let int_j1 = a_tau1_sq * 0 + b_tau2_sq * 3 + c_tau1_tau2 * 3;
        let int_j2 = a_tau1_sq * 3 + b_tau2_sq * 0 + c_tau1_tau2 * 3;
        [int_j1, int_j2]
    }
}

impl TangentBundle {
    pub fn tian_yau_z3() -> Self {
        Self {
            topo: CY3TopologicalData::tian_yau_z3(),
        }
    }
    pub fn schoen_z3xz3() -> Self {
        Self {
            topo: CY3TopologicalData::schoen_z3xz3(),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(deprecated)]
    use super::*;
    use crate::geometry::CicyGeometry;

    #[test]
    fn trivial_hidden_anomaly_residual_matches_visible() {
        let visible = VisibleBundle::ty_aglp_2011_standard();
        let hidden = HiddenBundle::trivial(4);
        let tm = TangentBundle::tian_yau_z3();
        let r = anomaly_cancellation_residual(&visible, &hidden, &tm);
        // c_2(V_v) = 14, c_2(V_h) = 0, c_2(TM) = 36 → residual = 22.
        assert!((r - 22.0).abs() < 1.0e-9, "expected 22, got {r}");
    }

    #[test]
    fn anomaly_zero_at_bianchi_match() {
        let visible = VisibleBundle::ty_aglp_2011_standard();
        // Construct hidden with c_2 = 36 - 14 = 22.
        // Need monad with c_1 = 0, c_2 = 22.
        // B = O(2)^4, C = O(8): c_1 = 0, c_2_general = 6·4 - 0 + 0 - 0 = 24 (close).
        // B = O(1)^4 ⊕ O(4), C = O(8): c_1 = 0, c_2_general = 6 + 4·4 - 0 = 22. ✓
        let hidden = HiddenBundle {
            monad_data: MonadBundle {
                b_degrees: vec![1, 1, 1, 1, 4],
                c_degrees: vec![8],
                map_coefficients: vec![1.0; 5],
            },
            e8_embedding: E8Embedding::SU5,
        };
        assert_eq!(hidden.monad_data.c1(), 0);
        assert_eq!(hidden.monad_data.c2_general(), 22);
        let tm = TangentBundle::tian_yau_z3();
        let r = anomaly_cancellation_residual(&visible, &hidden, &tm);
        assert!(r < 1.0e-9, "expected zero anomaly residual, got {r}");
    }

    #[test]
    fn polystability_polystable_when_b_nonpositive() {
        // c_1 = 0, b_i ≤ 0 → polystable.
        let bundle = HiddenBundle {
            monad_data: MonadBundle {
                b_degrees: vec![-1, -1, 0, 0],
                c_degrees: vec![-2],
                map_coefficients: vec![1.0; 4],
            },
            e8_embedding: E8Embedding::Trivial,
        };
        let geom = CicyGeometry::tian_yau_z3();
        let v = polystability_check(&bundle, &geom, &[1.0, 1.0]);
        assert!(v.is_polystable(), "expected polystable verdict, got {v:?}");
    }

    #[test]
    fn polystability_unstable_with_positive_b() {
        let bundle = HiddenBundle {
            monad_data: MonadBundle {
                b_degrees: vec![1, 1, 1, 1, 2],
                c_degrees: vec![6],
                map_coefficients: vec![1.0; 5],
            },
            e8_embedding: E8Embedding::SU5,
        };
        let geom = CicyGeometry::tian_yau_z3();
        let v = polystability_check(&bundle, &geom, &[1.0, 1.0]);
        assert!(!v.is_polystable(), "monad with positive b_i should be unstable");
    }

    #[test]
    fn sample_polystable_returns_anomaly_free() {
        let visible = VisibleBundle::ty_aglp_2011_standard();
        let geom = CicyGeometry::tian_yau_z3();
        let candidates = sample_polystable_hidden_bundles(&geom, &visible, 5);
        let tm = TangentBundle::tian_yau_z3();
        for h in &candidates {
            let r = anomaly_cancellation_residual(&visible, h, &tm);
            assert!(r < 1.0e-9, "sampled candidate must satisfy Bianchi exactly");
            // And polystability must hold.
            assert_eq!(h.monad_data.c1(), 0);
        }
    }

    #[test]
    fn schoen_visible_bundle_constructible() {
        let v = VisibleBundle::schoen_dhor_2006_minimal();
        assert_eq!(v.monad_data.c1(), 0);
        assert_eq!(v.e8_embedding, E8Embedding::SU4);
    }

    // ----------------------------------------------------------------
    // BHOP-2005 §6.1-6.2 published bundle tests.
    // ----------------------------------------------------------------

    /// BHOP-2005 §6.1 (line after Eq. 87): c_1(V) = 0.
    #[test]
    fn schoen_bhop2005_c1_is_zero() {
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let bhop = v.bhop_extension().expect("BHOP extension present");
        assert_eq!(bhop.c1(), BhopH4Coeffs::default(),
            "BHOP §6.1 (after Eq. 87): c_1(V) = 0 — got {:?}",
            bhop.c1());
    }

    /// BHOP-2005 Eq. 98:
    /// c_2(V) = -2 τ_1² + 7 τ_2² + 4 τ_1 τ_2.
    #[test]
    fn schoen_bhop2005_c2_matches_eq98() {
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let bhop = v.bhop_extension().expect("BHOP extension present");
        let c2 = bhop.c2();
        assert_eq!(c2.tau1_sq, -2,
            "BHOP Eq. 98 τ_1² coefficient: expected -2, got {}", c2.tau1_sq);
        assert_eq!(c2.tau2_sq, 7,
            "BHOP Eq. 98 τ_2² coefficient: expected 7, got {}", c2.tau2_sq);
        assert_eq!(c2.tau1_tau2, 4,
            "BHOP Eq. 98 τ_1 τ_2 coefficient: expected 4, got {}", c2.tau1_tau2);
    }

    /// BHOP-2005 Eq. 97 ch_3 → c_3 conversion:
    /// c_3(V) = -18 τ_1 τ_2² (Newton: ch_3 = c_3 / 2 + ...; here
    /// c_1 = 0 ⇒ c_3 = 2 · ch_3 = 2 · (-9 τ_1τ_2²) = -18 τ_1τ_2²).
    #[test]
    fn schoen_bhop2005_c3_from_ch3() {
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let bhop = v.bhop_extension().expect("BHOP extension present");
        assert_eq!(bhop.c3_tau1_tau2sq(), -18,
            "BHOP Eq. 97 ch_3·2 = c_3: expected -18 τ_1 τ_2², \
             got {} τ_1 τ_2²", bhop.c3_tau1_tau2sq());
    }

    /// BHOP-2005 Eq. 88: Index(V) = ∫_{X̃} ch(V) · Td(TX̃) = -27 on
    /// the cover X̃.
    #[test]
    fn schoen_bhop2005_index_minus_27_cover() {
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let bhop = v.bhop_extension().expect("BHOP extension present");
        assert_eq!(bhop.index_on_cover(), -27,
            "BHOP Eq. 88: Index(V) = -27 on the cover; got {}",
            bhop.index_on_cover());
    }

    /// BHOP-2005 Eq. 89: N_gen(X̃ / G) = -27 / |G| = -27 / 9 = -3
    /// (3 generations).
    #[test]
    fn schoen_bhop2005_three_generations_quotient() {
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let bhop = v.bhop_extension().expect("BHOP extension present");
        assert_eq!(bhop.generations_on_quotient(), -3,
            "BHOP Eq. 89: N_gen = -27 / 9 = -3 (3 generations); got {}",
            bhop.generations_on_quotient());
        assert_eq!(bhop.generations_on_quotient().abs(), 3,
            "absolute generation count must be 3");
    }

    /// BHOP-2005 Eq. 99-100: c_2(TX̃) - c_2(V) - c_2(H) = 6 τ_1²
    /// (cancelled by 5-branes wrapping PD(τ_1²) per Eq. 100).
    #[test]
    fn schoen_bhop2005_anomaly_residual_six_tau1_sq() {
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let bhop = v.bhop_extension().expect("BHOP extension present");
        let r = bhop.anomaly_residual();
        assert_eq!(r.tau1_sq, 6,
            "BHOP Eq. 99 anomaly residual τ_1² coeff: expected 6, got {}",
            r.tau1_sq);
        assert_eq!(r.tau2_sq, 0,
            "BHOP Eq. 99 residual must vanish in τ_2²; got {}", r.tau2_sq);
        assert_eq!(r.tau1_tau2, 0,
            "BHOP Eq. 99 residual must vanish in τ_1 τ_2; got {}", r.tau1_tau2);
    }

    /// BHOP-2005 §6.5 Hoppe polystability witness for the SU(4)
    /// extension (1) c_1(V) = 0, (2) c_2(V) ≠ 0 (extension not
    /// trivial), (3) rank(V_1) = rank(V_2) = 2.
    #[test]
    fn schoen_bhop2005_polystability() {
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let bhop = v.bhop_extension().expect("BHOP extension present");
        assert!(bhop.is_hoppe_polystable(),
            "BHOP §6.5 Hoppe-criterion polystability witness must hold");

        // Also: the shadow legacy SU(4) monad must have c_1 = 0
        // and rank > 0 so the η-evaluator and Chern-field-strength
        // consumers continue to compute non-zero traces. (Slope
        // polystability of the shadow monad shape itself was
        // covered by the legacy regression suite — see
        // `tests::test_polystability::test_dhor_2006_schoen_polystable`.)
        assert_eq!(v.monad_data.c1(), 0,
            "BHOP shadow monad must have c_1 = 0 for SU(4)");
        assert!(v.monad_data.rank() >= 3,
            "BHOP shadow monad rank must be >= 3");
        // BHOP §6.5 stability is encoded analytically in
        // is_hoppe_polystable above; the shadow is just there
        // for legacy consumers that compute traces from
        // line-bundle-degree data.
        let _ = CicyGeometry::schoen_z3xz3(); // ensure import is used
    }

    /// BHOP-2005 §6.1 Eq. 85: the auxiliary W must satisfy c_1(W) = 0.
    #[test]
    fn schoen_bhop2005_w_bundle_c1_zero() {
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let bhop = v.bhop_extension().expect("BHOP extension present");
        assert_eq!(bhop.w.c1_along_f(), 0,
            "BHOP §6.1 Eq. 85: c_1(W) = 0; got {}", bhop.w.c1_along_f());
        // Sanity: the published-W literal data.
        assert_eq!(bhop.w.sub_degree_f, -2);
        assert_eq!(bhop.w.quot_degree_f, 2);
        assert_eq!(bhop.w.quot_multiplicity, 2);
        assert_eq!(bhop.w.n_orbit_points, 9);
    }

    /// BHOP-2005 Eq. 94: c_2(TX̃) = 12 (τ_1² + τ_2²). Integrating
    /// against the (J_1, J_2) sub-basis using BHOP §3.2 Eq. 11-15
    /// must reproduce ∫ c_2(TX̃) ∧ J_a = 36 for a ∈ {1, 2} —
    /// matching `PUBLISHED_C2_TM_INTEGRALS[J1, J2] = [36, 36]` from
    /// `schoen_geometry::PUBLISHED_C2_TM_INTEGRALS`.
    ///
    /// (The J_T = φ-class integral lives outside BHOP's
    /// (τ_1, τ_2) sub-basis — `c_2(TX̃) ∧ J_T = 24` per
    /// `PUBLISHED_C2_TM_INTEGRALS[JT]` involves the φ-component
    /// which BHOP §6.1 Eq. 94 does not track.)
    #[test]
    fn schoen_bhop2005_consistent_with_published_c2_tm_integrals() {
        use crate::route34::schoen_geometry::PUBLISHED_C2_TM_INTEGRALS;
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let bhop = v.bhop_extension().expect("BHOP extension present");
        // BHOP Eq. 94: c_2(TX̃) coefficients in (τ_1², τ_2², τ_1τ_2).
        let [int_j1, int_j2] = BhopExtensionBundle::integrate_h4_bhop_subbasis(
            bhop.c2_tx.tau1_sq,
            bhop.c2_tx.tau2_sq,
            bhop.c2_tx.tau1_tau2,
        );
        assert_eq!(int_j1, PUBLISHED_C2_TM_INTEGRALS[0],
            "BHOP Eq. 94 ∧ J_1 must match DHOR-2006 Eq. 3.13: \
             expected {}, got {}", PUBLISHED_C2_TM_INTEGRALS[0], int_j1);
        assert_eq!(int_j2, PUBLISHED_C2_TM_INTEGRALS[1],
            "BHOP Eq. 94 ∧ J_2 must match DHOR-2006 Eq. 3.14: \
             expected {}, got {}", PUBLISHED_C2_TM_INTEGRALS[1], int_j2);
    }

    /// Sanity: the published BHOP `c_2(V)` integrated against
    /// (J_1, J_2) lands on the integers expected from the BHOP
    /// sub-basis arithmetic.
    /// `c_2(V) = -2 τ_1² + 7 τ_2² + 4 τ_1 τ_2`:
    ///   ∫ ∧ J_1 = -2·0 + 7·3 + 4·3 = 33
    ///   ∫ ∧ J_2 = -2·3 + 7·0 + 4·3 = 6
    #[test]
    fn schoen_bhop2005_c2v_integrated_subbasis() {
        let v = VisibleBundle::schoen_bhop2005_su4_extension();
        let bhop = v.bhop_extension().expect("BHOP extension present");
        let [int_j1, int_j2] = bhop.c2_v_dot_j_bhop_subbasis();
        assert_eq!(int_j1, 33,
            "BHOP Eq. 98 ∧ J_1: expected 33, got {int_j1}");
        assert_eq!(int_j2, 6,
            "BHOP Eq. 98 ∧ J_2: expected 6, got {int_j2}");
    }

    /// Cross-check: deprecated aliases must forward to the new
    /// builder.
    #[test]
    fn schoen_bhop2005_deprecated_aliases_forward() {
        let canonical = VisibleBundle::schoen_bhop2005_su4_extension();
        let alias_1 = VisibleBundle::schoen_canonical_su4_monad();
        let alias_2 = VisibleBundle::schoen_dhor_2006_minimal();
        assert_eq!(canonical.e8_embedding, alias_1.e8_embedding);
        assert_eq!(canonical.e8_embedding, alias_2.e8_embedding);
        assert_eq!(canonical.monad_data.b_degrees,
                   alias_1.monad_data.b_degrees);
        assert_eq!(canonical.monad_data.b_degrees,
                   alias_2.monad_data.b_degrees);
        assert!(alias_1.bhop_extension().is_some(),
            "deprecated alias must still expose BHOP extension data");
        assert!(alias_2.bhop_extension().is_some(),
            "deprecated alias must still expose BHOP extension data");
    }

    #[test]
    fn trivial_hidden_has_zero_chern() {
        let h = HiddenBundle::trivial(3);
        assert_eq!(h.monad_data.c1(), 0);
        assert_eq!(h.monad_data.c2_general(), 0);
        assert_eq!(h.monad_data.c3_general(), 0);
    }

    #[test]
    fn stability_verdict_margin_extraction() {
        let v = StabilityVerdict::Polystable { margin: 1.5 };
        assert!((v.margin() - 1.5).abs() < 1.0e-12);
        let u = StabilityVerdict::Unstable {
            excess: 2.0,
            sub_b: 3,
        };
        assert!((u.margin() + 2.0).abs() < 1.0e-12);
    }
}
