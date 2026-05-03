//! Top-level Route 4 predictor: combines Arnold (Step 2) + Rossby
//! (Step 3) + Killing-spectrum (Step 4) into a discrimination score
//! against the observed Saturn/Jupiter polar wavenumbers
//! `{Saturn n=6, Jupiter north n=8, Jupiter south n=5}`.
//!
//! ## Discrimination score (chapter-21 hard-gating semantics)
//!
//! Per the chapter-21 commitment (line 277 of
//! `08-choosing-a-substrate.adoc`), the candidate-CY3's Killing-vector
//! spectrum projected through Arnold's classification at the polar
//! critical boundary defines the SET of admissible wavenumbers. The
//! observed wavenumber is either in that set (the candidate matches)
//! or not (the candidate is incompatible).
//!
//! We implement this as a hard-gated discrete-distance χ²:
//!
//! ```text
//!   admissible(state, killing) = arnold_set(state) ∩ killing_set(killing)
//!   d_i = min_{s ∈ admissible} |obs_i − s|
//!   χ² = Σ_i d_i²
//! ```
//!
//! When `obs_i` is in the admissible set, `d_i = 0` and the
//! contribution to χ² is exactly zero. When it is not in the set,
//! `d_i ≥ 1` and the χ² contribution is bounded below by 1.
//!
//! The intersection of (a) the Arnold-classified set (Coxeter
//! exponents of the polar Lyapunov germ's ADE type) and (b) the
//! Killing-algebra-derived set (multiples of cyclic-subgroup orders
//! plus the continuous-isometry-dimension allowance) is what makes
//! the Killing spectrum actually drive the Route 4 discrimination —
//! a candidate's cyclic factors must contain the right divisors of
//! the observed wavenumber (or admit it via continuous isometry)
//! AND the Lyapunov germ at the polar critical boundary must be of
//! an ADE type whose Coxeter exponents include it.
//!
//! ## What this module is NOT
//!
//! This is NOT a falsification verdict on its own. The polyhedral-
//! resonance mapping (substrate-amplitude wavenumber ↔ ADE Coxeter
//! exponent at polar critical boundary) is a framework conjecture;
//! the discrete χ² is a hard-gated discrimination signal between
//! candidate CY3s, but the Bayesian likelihood layer combines it
//! with Routes 1-3 before pronouncing a verdict.
//!
//! References:
//!   - chapter-21 §"Pinning Down Route 4" lines 265-298 of
//!     `book/chapters/part3/08-choosing-a-substrate.adoc`.
//!   - Bourbaki, "Groupes et algèbres de Lie", Ch. VI §1.11
//!     (Coxeter exponents).
//!   - Arnold, *Russian Math. Surveys* **29** (1974) 10,
//!     DOI 10.1070/RM1974v029n02ABEH002889.
//!   - As also referenced in [`crate::route34::arnold_normal_form`]
//!     and [`crate::route34::rossby_polar`].

use crate::route34::arnold_normal_form::{
    admissible_wavenumber_set, classify_singularity, ArnoldType, GermError,
};
use crate::route34::rossby_polar::{
    arnold_type_cutoff, linearised_lyapunov, published_jupiter_north_polar,
    published_jupiter_south_polar, published_saturn_polar, PolarBasicState,
};
use crate::route34::{CyclicSubgroup, KillingResult};

/// Output of the Route 4 discrimination computation.
///
/// Hard-gated semantics (chapter-21 line 277): each `*_match` field
/// is `1.0` when the observed wavenumber is in the candidate's
/// admissible set (Arnold-set ∩ Killing-set) and `0.0` otherwise.
/// The `*_distance` fields record the discrete distance to the
/// nearest admissible wavenumber (zero when in set, ≥ 1 otherwise).
#[derive(Debug, Clone)]
pub struct Route4Prediction {
    /// Hard match for Saturn (n=6 observed): 1.0 if 6 ∈ admissible, else 0.
    pub saturn_match: f64,
    /// Hard match for Jupiter north (n=8 observed).
    pub jupiter_north_match: f64,
    /// Hard match for Jupiter south (n=5 observed).
    pub jupiter_south_match: f64,
    /// `min_{s ∈ admissible} |6 − s|`.
    pub saturn_distance: u32,
    pub jupiter_north_distance: u32,
    pub jupiter_south_distance: u32,
    /// χ² = Σ d_i² (discrete distances). Zero ⇔ all observed
    /// wavenumbers are in the admissible set.
    pub combined_chi_squared: f64,
    /// The full admissible wavenumber sets for each regime
    /// (Arnold-set ∩ Killing-set) — useful for downstream report
    /// formatting and Bayesian likelihood inputs.
    pub saturn_predicted: Vec<u32>,
    pub jupiter_north_predicted: Vec<u32>,
    pub jupiter_south_predicted: Vec<u32>,
    /// The Arnold ADE classification of each polar Lyapunov germ.
    pub saturn_arnold_type: ArnoldType,
    pub jupiter_north_arnold_type: ArnoldType,
    pub jupiter_south_arnold_type: ArnoldType,
    /// The candidate's label, copied from the input `KillingResult`.
    pub candidate_label: String,
}

/// Wavenumber set induced by the candidate's Killing spectrum alone
/// (no Arnold projection). Multiples of each cyclic-subgroup order
/// (up to `cutoff`) plus the continuous-isometry-dimension allowance
/// `1..=continuous_isometry_dim`.
///
/// This is the "Killing-set" half of the chapter-21 admissibility
/// intersection. Returned values are sorted ascending and deduped.
///
/// The trivial wavenumber `1` (constant mode) is always included.
pub fn killing_admissible_wavenumbers(
    killing_subgroups: &[CyclicSubgroup],
    continuous_isometry_dim: u32,
    cutoff: u32,
) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::new();
    out.push(1);
    if continuous_isometry_dim > 0 {
        for n in 1..=continuous_isometry_dim {
            out.push(n);
        }
    }
    for sub in killing_subgroups {
        let n = sub.order;
        if n <= 1 {
            continue;
        }
        let mut k: u32 = 1;
        loop {
            let v = n * k;
            if v > cutoff {
                break;
            }
            out.push(v);
            k += 1;
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Admissible wavenumber set at a polar critical-boundary regime,
/// as the intersection of the Arnold-classified set (Coxeter exponents
/// of the local Lyapunov germ's ADE type) with the Killing-spectrum-
/// derived set (multiples of cyclic-subgroup orders + continuous-
/// isometry-dimension allowance).
///
/// This is the chapter-21 hard-gating semantics: a wavenumber is
/// admissible iff it is BOTH a Coxeter exponent of the local Arnold
/// type AND compatible with the candidate's Killing spectrum.
///
/// Returns `(admissible_set, arnold_type)` so the caller can
/// distinguish the Arnold classification from the gated set.
///
/// Reference: chapter-21 line 277 of
/// `book/chapters/part3/08-choosing-a-substrate.adoc`.
pub fn admissible_wavenumbers_from_killing_and_arnold(
    state: &PolarBasicState,
    killing_subgroups: &[CyclicSubgroup],
    continuous_isometry_dim: u32,
    perturbation_basis_dim: usize,
) -> Result<(Vec<u32>, ArnoldType), GermError> {
    let germ = linearised_lyapunov(state, perturbation_basis_dim)?;
    let arnold_type = classify_singularity(&germ)?;
    let arnold_set = admissible_wavenumber_set(arnold_type);
    let cutoff = arnold_type_cutoff(arnold_type);
    let killing_set = killing_admissible_wavenumbers(
        killing_subgroups,
        continuous_isometry_dim,
        cutoff,
    );
    // Set intersection. We build it manually rather than via HashSet
    // to keep deterministic ordering (Arnold-set order preserved).
    let mut gated: Vec<u32> = arnold_set
        .iter()
        .copied()
        .filter(|n| killing_set.contains(n))
        .collect();
    gated.sort_unstable();
    gated.dedup();
    Ok((gated, arnold_type))
}

/// Top-level Route 4 entry point. Given a candidate CY3's
/// Killing-spectrum result, predict Saturn/Jupiter polar wavenumbers
/// (under the chapter-21 hard-gating semantics) and return the
/// discrimination score.
pub fn route4_discrimination(
    candidate_killing: &KillingResult,
) -> Result<Route4Prediction, GermError> {
    let saturn = published_saturn_polar();
    let jup_n = published_jupiter_north_polar();
    let jup_s = published_jupiter_south_polar();

    // Each planet's polar Lyapunov germ is built from its own basic-
    // state parameters, classified by the Arnold engine, and gated
    // by the candidate's Killing-spectrum admissibility set.
    let (saturn_set, saturn_type) =
        admissible_wavenumbers_from_killing_and_arnold(
            &saturn,
            &candidate_killing.cyclic_factors,
            candidate_killing.continuous_isometry_dim,
            2,
        )?;
    let (jup_n_set, jup_n_type) =
        admissible_wavenumbers_from_killing_and_arnold(
            &jup_n,
            &candidate_killing.cyclic_factors,
            candidate_killing.continuous_isometry_dim,
            2,
        )?;
    let (jup_s_set, jup_s_type) =
        admissible_wavenumbers_from_killing_and_arnold(
            &jup_s,
            &candidate_killing.cyclic_factors,
            candidate_killing.continuous_isometry_dim,
            2,
        )?;

    let (m_saturn, d_saturn) = hard_gate_match(6, &saturn_set);
    let (m_jup_n, d_jup_n) = hard_gate_match(8, &jup_n_set);
    let (m_jup_s, d_jup_s) = hard_gate_match(5, &jup_s_set);

    // χ² = Σ d_i² (discrete distance from observed to nearest
    // admissible). Zero iff all three observed wavenumbers are in the
    // admissible (Arnold ∩ Killing) sets.
    let chi2 = (d_saturn as f64).powi(2)
        + (d_jup_n as f64).powi(2)
        + (d_jup_s as f64).powi(2);

    Ok(Route4Prediction {
        saturn_match: m_saturn,
        jupiter_north_match: m_jup_n,
        jupiter_south_match: m_jup_s,
        saturn_distance: d_saturn,
        jupiter_north_distance: d_jup_n,
        jupiter_south_distance: d_jup_s,
        combined_chi_squared: chi2,
        saturn_predicted: saturn_set,
        jupiter_north_predicted: jup_n_set,
        jupiter_south_predicted: jup_s_set,
        saturn_arnold_type: saturn_type,
        jupiter_north_arnold_type: jup_n_type,
        jupiter_south_arnold_type: jup_s_type,
        candidate_label: candidate_killing.candidate_label.clone(),
    })
}

/// Hard-gate match: returns `(1.0, 0)` if `obs ∈ admissible`, else
/// `(0.0, d)` where `d = min_{s ∈ admissible} |obs − s|`.
///
/// If `admissible` is empty, returns `(0.0, obs)` — the observed
/// wavenumber's full magnitude as the discrete distance penalty.
fn hard_gate_match(obs: u32, admissible: &[u32]) -> (f64, u32) {
    if admissible.contains(&obs) {
        return (1.0, 0);
    }
    if admissible.is_empty() {
        return (0.0, obs);
    }
    let d = admissible
        .iter()
        .map(|&p| {
            let diff = (p as i64) - (obs as i64);
            diff.unsigned_abs() as u32
        })
        .min()
        .unwrap_or(obs);
    (0.0, d)
}

/// Convenience: run the predictor for the four canonical
/// candidates [`KillingResult::polysphere_s3xs3`],
/// [`KillingResult::flat_t6`], [`KillingResult::tianyau_z3`],
/// [`KillingResult::schoen_z3xz3`]. Useful for quick discrimination
/// reports.
pub fn route4_canonical_candidates() -> Result<Vec<Route4Prediction>, GermError> {
    let candidates = [
        KillingResult::polysphere_s3xs3(),
        KillingResult::flat_t6(),
        KillingResult::tianyau_z3(),
        KillingResult::schoen_z3xz3(),
        KillingResult::generic_no_isometry(),
    ];
    let mut out = Vec::with_capacity(candidates.len());
    for c in &candidates {
        out.push(route4_discrimination(c)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hard_gate_match_basic() {
        // obs in set ⇒ (1.0, 0).
        assert_eq!(hard_gate_match(6, &[1, 2, 6, 8]), (1.0, 0));
        // obs not in set ⇒ (0.0, distance to nearest).
        assert_eq!(hard_gate_match(7, &[1, 2, 6, 8]), (0.0, 1));
        assert_eq!(hard_gate_match(4, &[1, 7, 9]), (0.0, 3));
        // empty set ⇒ (0.0, obs).
        assert_eq!(hard_gate_match(5, &[]), (0.0, 5));
    }

    #[test]
    fn test_killing_admissible_wavenumbers_z3() {
        // A candidate with a single Z/3 factor should produce
        // admissible wavenumbers {1, 3, 6, 9, ...} up to cutoff.
        let killing = vec![CyclicSubgroup::new(3)];
        let set = killing_admissible_wavenumbers(&killing, 0, 12);
        assert!(set.contains(&1));
        assert!(set.contains(&3));
        assert!(set.contains(&6));
        assert!(set.contains(&9));
        assert!(set.contains(&12));
        // Wavenumber 5 not a multiple of 3 ⇒ should NOT be admissible.
        assert!(!set.contains(&5));
        // Wavenumber 8 not a multiple of 3 ⇒ should NOT be admissible.
        assert!(!set.contains(&8));
    }

    #[test]
    fn test_killing_admissible_wavenumbers_continuous_isometry() {
        // Continuous isometry dim 12 admits wavenumbers 1..=12 as
        // additional resonances on top of any cyclic-subgroup
        // multiples.
        let killing = vec![CyclicSubgroup::trivial()];
        let set = killing_admissible_wavenumbers(&killing, 12, 12);
        for n in 1..=12u32 {
            assert!(set.contains(&n), "missing wavenumber {n}");
        }
    }

    #[test]
    fn test_route4_runs_on_all_canonical_candidates() {
        let preds = route4_canonical_candidates().unwrap();
        assert_eq!(preds.len(), 5);
        for p in &preds {
            // Hard-gate matches are exactly 0.0 or 1.0.
            assert!(p.saturn_match == 0.0 || p.saturn_match == 1.0);
            assert!(
                p.jupiter_north_match == 0.0 || p.jupiter_north_match == 1.0
            );
            assert!(
                p.jupiter_south_match == 0.0 || p.jupiter_south_match == 1.0
            );
            assert!(p.combined_chi_squared >= 0.0);
            // χ² is the sum of squared discrete distances; bounded
            // above by 6² + 8² + 5² = 125 in the worst case.
            assert!(p.combined_chi_squared <= 125.0 + 1.0e-9);
        }
    }

    #[test]
    fn test_killing_spectrum_changes_admissible_wavenumbers() {
        // The chapter-21 hard-gating semantics is INTERSECTION of the
        // Arnold-classified set ((Coxeter exponents) ∪ {Coxeter number}
        // of the local Lyapunov germ's ADE type) with the Killing-
        // spectrum-derived set. Changing the cyclic factors changes
        // the intersection ⇒ the predicted set changes.
        //
        // The Saturn polar germ at q_coef=0 classifies as D_4
        // hyperbolic umbilic with admissible set
        //   (exponents {1, 3, 5}) ∪ (Coxeter number {6}) = {1, 3, 5, 6}
        // (Bourbaki Ch VI §1.11; Humphreys §3.18; h(D_4) = 2(4-1) = 6).
        // - With a Z/3 cyclic factor: Killing-set is {1, 3, 6, 9, …}
        //   (multiples of 3 up to cutoff). Intersection with
        //   {1, 3, 5, 6} is {1, 3, 6}.
        // - With Z/5 instead: Killing-set is {1, 5, 10, …}.
        //   Intersection is {1, 5}.
        // So the Killing spectrum genuinely changes which D_4
        // admissible modes survive the gating; under Z/3 the Coxeter-
        // number contribution n=6 IS preserved (6 is a multiple of 3),
        // while under Z/5 it is not.
        let saturn_state = published_saturn_polar();
        let with_z3 = vec![CyclicSubgroup::new(3)];
        let with_z5 = vec![CyclicSubgroup::new(5)];

        let (set_z3, _) = admissible_wavenumbers_from_killing_and_arnold(
            &saturn_state,
            &with_z3,
            0,
            2,
        )
        .unwrap();
        let (set_z5, _) = admissible_wavenumbers_from_killing_and_arnold(
            &saturn_state,
            &with_z5,
            0,
            2,
        )
        .unwrap();

        // Both sets contain the trivial wavenumber 1.
        assert!(set_z3.contains(&1));
        assert!(set_z5.contains(&1));
        // Z/3 candidate admits 3 (multiple of 3 AND a D_4 Coxeter
        // exponent); Z/5 candidate does not (3 is in D_4 exponents
        // but not in {1, 5, 10, …}).
        assert!(
            set_z3.contains(&3),
            "Z/3 set should contain 3 via D_4 admissible ∩ {{3, 6, 9, …}}; got {set_z3:?}"
        );
        assert!(
            !set_z5.contains(&3),
            "Z/5 set should NOT contain 3 (Killing-set is {{1, 5, 10, …}}); got {set_z5:?}"
        );
        // Z/5 candidate admits 5; Z/3 candidate does not.
        assert!(set_z5.contains(&5), "Z/5 set should contain 5; got {set_z5:?}");
        assert!(!set_z3.contains(&5), "Z/3 set should NOT contain 5; got {set_z3:?}");
        // Saturn n=6 is in the D_4 admissible set (Coxeter number
        // h(D_4) = 6). Under Z/3 gating 6 is preserved (6 = 2·3),
        // under Z/5 gating 6 is filtered out (6 is not a multiple
        // of 5). This is the principled basis for Route 4 to
        // discriminate Saturn under the chapter-21 hard-gating
        // semantics.
        assert!(
            set_z3.contains(&6),
            "Z/3 set should contain 6 (D_4 Coxeter number h=6, and 6 = 2·3 ∈ Z/3 multiples); got {set_z3:?}"
        );
        assert!(
            !set_z5.contains(&6),
            "Z/5 set should NOT contain 6 (6 is not a multiple of 5); got {set_z5:?}"
        );
        // Therefore the two sets differ ⇒ the Killing spectrum
        // genuinely drives the admissible-wavenumber prediction.
        assert_ne!(
            set_z3, set_z5,
            "the gated admissible sets must differ between Z/3 and Z/5 \
             candidates; both came out as {set_z3:?}"
        );

        // Hard-gate distance for n=6: zero under Z/3 (admissible),
        // nonzero under Z/5 (filtered out).
        let (m_z3, d_z3) = hard_gate_match(6, &set_z3);
        let (m_z5, d_z5) = hard_gate_match(6, &set_z5);
        assert_eq!(
            (m_z3, d_z3),
            (1.0, 0),
            "Saturn n=6 should be admissible under Z/3 D_4 gating; got match={m_z3} dist={d_z3}"
        );
        assert!(
            d_z5 > 0 && m_z5 == 0.0,
            "Saturn n=6 should NOT be admissible under Z/5 D_4 gating; got match={m_z5} dist={d_z5}"
        );
    }

    #[test]
    fn integration_polysphere_matches_saturn_and_jupiter_south_only() {
        // Integration check under chapter-21 hard-gating semantics:
        // the polar Lyapunov germ classifies as D_4 by construction
        // (q_coef = 0 commitment in rossby_polar.rs:316), so the
        // Arnold-set is {1, 3, 5} ∪ {h(D_4) = 6} = {1, 3, 5, 6}
        // regardless of Killing structure. The intersection with any
        // candidate's Killing-set therefore can never include n=7 or
        // n=8.
        //
        // Saturn n=6 and Jupiter south n=5 are admissible under any
        // candidate whose Killing-set contains {5, 6}; here we use a
        // Z/3 + continuous-dim 8 candidate which trivially does.
        // Jupiter north n=8 is NOT admissible under D_4 — the chapter
        // explicitly leaves Jupiter-N n=8 as an open question per
        // §"Bottom Line and Next Step" (lines 299-306).
        let killing = KillingResult {
            candidate_label: "polysphere-with-z3-test".to_string(),
            cyclic_factors: vec![CyclicSubgroup::new(3)],
            continuous_isometry_dim: 8,
        };
        let pred = route4_discrimination(&killing).unwrap();
        // Saturn n=6: D_4 admissible includes h(D_4)=6.
        assert_eq!(
            pred.saturn_match, 1.0,
            "Saturn n=6 should match under (Z/3, dim≥6) candidate; predicted set: {:?}",
            pred.saturn_predicted
        );
        // Jupiter south n=5: D_4 admissible includes exponent 5.
        assert_eq!(
            pred.jupiter_south_match, 1.0,
            "Jupiter n=5 should match; predicted set: {:?}",
            pred.jupiter_south_predicted
        );
        // Jupiter north n=8: NOT in D_4 admissible — open per chapter.
        // Closest admissible is 6 (distance 2), so χ² = 2² = 4.
        assert_eq!(
            pred.jupiter_north_match, 0.0,
            "Jupiter n=8 should NOT match D_4 polar germ; predicted set: {:?}",
            pred.jupiter_north_predicted
        );
        assert_eq!(pred.jupiter_north_distance, 2);
        assert_eq!(pred.combined_chi_squared, 4.0);
    }

    #[test]
    fn jupiter_n5_still_admissible_under_d4() {
        // Sanity check that the Coxeter-number extension is purely
        // additive: Jupiter south n=5 (a D_4 Coxeter exponent) must
        // still pass under a Z/5 candidate — 5 was admissible
        // pre-extension and the Coxeter-number addition cannot remove
        // it.
        let saturn_state = published_saturn_polar();
        let with_z5 = vec![CyclicSubgroup::new(5)];
        let (set_z5, _) = admissible_wavenumbers_from_killing_and_arnold(
            &saturn_state,
            &with_z5,
            0,
            2,
        )
        .unwrap();
        assert!(
            set_z5.contains(&5),
            "Jupiter n=5 (D_4 Coxeter exponent) must remain admissible under Z/5 gating; got {set_z5:?}"
        );
    }

    #[test]
    fn test_ty_vs_schoen_produce_well_defined_predictions() {
        let ty = route4_discrimination(&KillingResult::tianyau_z3()).unwrap();
        let sch =
            route4_discrimination(&KillingResult::schoen_z3xz3()).unwrap();
        assert_ne!(ty.candidate_label, sch.candidate_label);
        assert!(ty.combined_chi_squared.is_finite());
        assert!(sch.combined_chi_squared.is_finite());
    }
}
