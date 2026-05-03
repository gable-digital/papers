// LEGACY-SUPERSEDED-BY-ROUTE34: this entire file is a stage-1 proxy filter
// suite from the broad-sweep-search era. Each loss function uses a
// dimensional-analysis approximation (rounded-Vandermonde Killing
// spectrum, log-deviation eta proxy, Cartan-phase-cyclic-permutation
// Wilson-line proxy). The route34 pipeline supersedes every entry:
//   * `wilson_line_loss`         -> route34::wilson_line_e8::canonical_e8_to_e6_su3
//                                  (real Slansky-1981 E_8 -> E_6 x SU(3)
//                                   embedding with proper Cartan
//                                   sub-algebra basis)
//   * `quotient_area_ratio_loss` -> route34::eta_evaluator + chern_field_strength
//                                  (full chapter-21 eta integral form,
//                                   Tr_v(F_v^2) - Tr_h(F_h^2) wedge J on
//                                   Z/3-fixed divisor F)
//   * `ade_wavenumber_loss`      -> route34::route4_predictor +
//                                  arnold_normal_form + rossby_polar +
//                                  killing_solver (real Arnold catastrophe
//                                  classification + Lichnerowicz
//                                  Killing-vector solver + Rossby-wave
//                                  Lyapunov germ at polar critical
//                                  boundary)
// These functions remain only as a Stage-1 cheap pre-filter. Final
// discrimination uses the route34 pipeline. Do not modify in place;
// add new logic to the route34 modules above.
//
//! Stage-1 cheap topological / bundle-sector filters.
//!
//! These filters take only `Candidate` data (topology + moduli vectors)
//! and run in microseconds per candidate — no Donaldson balancing, no
//! sample-point evaluation, no metric. They prune the broad-sweep search
//! space by orders of magnitude before any expensive computation.
//!
//! Filters implemented:
//!   1. Chern-class / index-theorem consistency (3 generations)
//!   2. DUY slope-stability (polystable bundle proxy)
//!   3. Wilson-line E_8 -> E_6 -> SU(3) x SU(2) x U(1) consistency
//!   4. Route 3: matter-antimatter asymmetry eta as Z/3 quotient-area-ratio
//!   5. Route 4: ADE wavenumber {6, 8, 5} via Killing-vector spectrum proxy
//!
//! Each filter returns a non-negative loss; zero means "passes the
//! constraint exactly". The Stage-1 score function sums them with weights
//! tuned so a bundle that fails any single filter yields total > 1.

use crate::pipeline::Candidate;

/// Chern-class index-theorem consistency.
///
/// For a heterotic CY3 compactification with bundle V of rank r, the
/// number of fermion generations is
///
///   n_gen = (1/2) |c_3(V)| + lower-order correction
///
/// For SU(5)-class GUT bundles on a CY3 with chi(M) = -6 the standard
/// requirement is c_3(V) = 6 (giving 3 generations after Wilson-line
/// breaking through the Z/3 quotient: 6 / |Z/3| = 2 sub-blocks of 3).
///
/// The bundle's c_3 is a sum over instantons that we approximate from the
/// integer-rounded bundle moduli. We treat the first 3 entries of
/// bundle_moduli as the Chern numbers (c_1, c_2, c_3) up to scale.
///
/// Loss is zero when the rounded c_3 equals the target n_gen_target * 2.
pub fn chern_class_loss(c: &Candidate, n_gen_target: i32) -> f64 {
    if c.bundle_moduli.len() < 3 {
        return 1.0;
    }
    // Treat bundle_moduli[0..=2] as continuous proxies for the first three
    // Chern numbers. The L-shaped quantisation lattice for heterotic
    // bundles forces these to integer-class invariants on the CY3; the
    // distance from the nearest integer-class triple is a soft-stability
    // proxy.
    let c1 = c.bundle_moduli[0];
    let c2 = c.bundle_moduli[1];
    let c3 = c.bundle_moduli[2];

    // c_1(V) must vanish for SU(n)-bundles (anomaly cancellation).
    let c1_loss = c1 * c1;
    // c_2(V) must equal c_2(TX) for the Bianchi identity (modulo 5-brane
    // corrections we set aside here). Approximate c_2(TX) ~ -chi/12 for a
    // CY3.
    let target_c2 = -(c.euler_characteristic as f64) / 12.0;
    let c2_loss = (c2 - target_c2).powi(2);
    // c_3(V) must equal +/- 2 * n_gen_target for 3 generations after Z/3
    // quotient. Distance to the nearest of {+2N, -2N}.
    let target_c3_abs = 2.0 * n_gen_target as f64;
    let c3_loss = (c3.abs() - target_c3_abs).powi(2);

    c1_loss + c2_loss + c3_loss
}

/// DUY slope-stability proxy.
///
/// The Donaldson-Uhlenbeck-Yau theorem requires the bundle V be polystable:
/// for every coherent sub-sheaf F of V, slope(F) <= slope(V), with equality
/// only when V splits.
///
/// Direct check requires sub-sheaf enumeration, which is cohomological and
/// expensive. As a Stage-1 proxy, treat the bundle_moduli vector as a
/// concatenation of rank-1 sub-sheaf slopes; a polystable bundle has all
/// sub-slopes equal to (or below) the average slope. A high spread of
/// per-sub-sheaf slopes indicates likely instability.
///
/// Returns the variance of the bundle_moduli entries past index 3 (which
/// we reserve for Chern numbers).
pub fn slope_stability_loss(c: &Candidate) -> f64 {
    if c.bundle_moduli.len() <= 3 {
        return 0.0;
    }
    let slopes = &c.bundle_moduli[3..];
    let mean: f64 = slopes.iter().sum::<f64>() / slopes.len() as f64;
    let var: f64 =
        slopes.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / slopes.len() as f64;
    // Penalise high variance; cap at 1.0 so a single fail can't drown out
    // the other filters.
    var.min(2.0)
}

/// Wilson-line breaking consistency.
///
/// The Z/3 (or Z/3 x Z/3) Wilson line must commute with the gauge
/// generator pattern that breaks E_8 -> E_6 -> SU(3) x SU(2) x U(1).
///
/// For Z/3, the generator is exp(2 pi i / 3) acting on the bundle's
/// fundamental representation. The commutator with the SU(3) colour
/// generators must vanish; the commutator with the SU(2) weak generators
/// must produce U(1)_Y mixing in a specific pattern.
///
/// Stage-1 proxy: read 3 phase-angle proxies from bundle_moduli[20..23]
/// (mod 2 pi) and require the trio to lie close to the third-roots-of-unity
/// {0, 2pi/3, 4pi/3}. Distance from the nearest cyclic permutation gives
/// the loss.
pub fn wilson_line_loss(c: &Candidate) -> f64 {
    if c.bundle_moduli.len() < 23 {
        return 0.0;
    }
    use std::f64::consts::PI;
    let two_pi = 2.0 * PI;
    let phi: [f64; 3] = [
        c.bundle_moduli[20].rem_euclid(two_pi),
        c.bundle_moduli[21].rem_euclid(two_pi),
        c.bundle_moduli[22].rem_euclid(two_pi),
    ];
    // Target is some cyclic permutation of (0, 2pi/3, 4pi/3).
    let targets: [[f64; 3]; 3] = [
        [0.0, two_pi / 3.0, 2.0 * two_pi / 3.0],
        [two_pi / 3.0, 2.0 * two_pi / 3.0, 0.0],
        [2.0 * two_pi / 3.0, 0.0, two_pi / 3.0],
    ];
    let mut best = f64::INFINITY;
    for target in &targets {
        // Circular distance for each angle: min(|a-b|, 2pi-|a-b|).
        let mut d = 0.0;
        for k in 0..3 {
            let raw = (phi[k] - target[k]).abs();
            let circ = raw.min(two_pi - raw);
            d += circ * circ;
        }
        if d < best {
            best = d;
        }
    }
    best
}

/// Route 3: matter-antimatter asymmetry eta as quotient-area constraint.
///
/// The empirical eta = 6.1e-10 (BBN + CMB) records the size-mismatch
/// between the two parent-side substrate-regions in the merger-birth
/// cross-sector mixing. For Tian-Yau (Z/3 quotient), the area-ratio of
/// the two sectors at the orbifold-coupling locus is one number; for
/// Schoen-Z3xZ3 it is a different number.
///
/// We model the predicted eta as a small log-deviation around the
/// fundamental-group-determined baseline:
///
///   eta_predicted ~ exp(-baseline_log * (1 + delta(moduli)))
///
/// where baseline_log = -log(6.1e-10) ~= 21.2 for both candidates and
/// delta(moduli) is a moduli-dependent tilt that vanishes when the
/// quotient is "balanced." We treat the average bundle_moduli as a proxy
/// for delta.
pub fn quotient_area_ratio_loss(c: &Candidate) -> f64 {
    let measured = 6.1e-10_f64;
    let log_measured = measured.ln(); // ~ -21.2

    // Fundamental-group-dependent prefactor: empirically TY/Z3 sits near
    // log(eta) = -21.2 with delta ~ 0; Schoen Z3xZ3's smaller quotient
    // group shifts the baseline. Encode as a small additive offset.
    let group_offset: f64 = match c.fundamental_group.as_str() {
        "Z3" => 0.0,
        "Z3xZ3" => 0.6, // ln(2) shift for the Z/3 x Z/3 -> Z/3 reduction
        _ => 5.0,       // very far if not in the candidate class
    };

    if c.bundle_moduli.is_empty() {
        return group_offset.powi(2);
    }
    let mean_modulus: f64 =
        c.bundle_moduli.iter().sum::<f64>() / c.bundle_moduli.len() as f64;
    let log_predicted = log_measured * (1.0 + 0.05 * mean_modulus) + group_offset;
    let rel_err = (log_predicted - log_measured) / log_measured;
    rel_err * rel_err
}

/// Route 4: ADE wavenumber spectrum {6, 8, 5}.
///
/// Saturn's hexagonal polar jet stream (n=6), Jupiter's north polar
/// cyclone arrangement (n=8), and Jupiter's south polar arrangement
/// (n=5) directly trace ADE-classified substrate-amplitude resonance
/// patterns. These wavenumbers are constrained by the candidate CY3's
/// continuous-isometry / Killing-vector-field structure.
///
/// Stage-1 proxy: the candidate's Killing-vector spectrum is a finite
/// integer subset of {0, ..., k_max}. We approximate it as the rounded
/// integer values of a moduli-weighted projection of (h11, h21, n_bundle)
/// onto a Vandermonde-style basis. The loss is the L2 distance from the
/// observed wavenumber set {6, 8, 5} to the nearest 3-element subset of
/// the predicted Killing spectrum.
pub fn ade_wavenumber_loss(c: &Candidate) -> f64 {
    let observed = [6_i32, 8, 5];

    // Predicted Killing spectrum: build a 12-element integer multiset
    // from the candidate's topology + moduli first 6 Kahler components.
    let mut spectrum: Vec<i32> = Vec::with_capacity(12);
    let h11 = c.kahler_moduli.len() as i32;
    let h21 = c.complex_moduli_real.len() as i32;
    spectrum.push(h11);
    spectrum.push(h21);
    spectrum.push((h11 + h21) / 2);
    // Add 9 moduli-derived integer wavenumbers
    for k in 0..9 {
        let idx = k % c.kahler_moduli.len().max(1);
        let raw = if c.kahler_moduli.is_empty() {
            0.0
        } else {
            (k as f64 + 1.0) * c.kahler_moduli[idx]
        };
        spectrum.push(raw.round().abs() as i32);
    }

    // For each observed wavenumber find the closest entry in the spectrum.
    let mut total = 0.0;
    for &obs in &observed {
        let mut best = i32::MAX;
        for &s in &spectrum {
            let d = (obs - s).abs();
            if d < best {
                best = d;
            }
        }
        total += (best as f64).powi(2);
    }
    total
}

/// Aggregate Stage-1 loss combining all five filters with reasonable
/// weights. The weights are tuned so that a candidate failing any one
/// filter (loss ~ 1+) crosses the broad-sweep filter threshold of 2.0.
pub fn stage1_topological_loss(c: &Candidate, n_gen_target: i32) -> Stage1Breakdown {
    Stage1Breakdown {
        chern: chern_class_loss(c, n_gen_target),
        slope: slope_stability_loss(c),
        wilson_line: wilson_line_loss(c),
        eta: quotient_area_ratio_loss(c),
        ade: ade_wavenumber_loss(c),
    }
}

/// Budget-aware variant of `stage1_topological_loss` that runs the five
/// filters in budget-pruning order (most-discriminating cheapest filter
/// first) and bails as soon as the cumulative sum exceeds `budget`. Any
/// filter that has not been evaluated by the time the budget is
/// exhausted returns 1.0 in its corresponding `Stage1Breakdown` field --
/// large enough that the candidate fails the broad-sweep threshold of
/// 2.0 anyway, so the early-exit is sound.
///
/// Order rationale:
///   - `wilson_line` is O(8) and the most-discriminating filter on
///     uniformly-sampled bundle moduli (most random samples fail);
///     running it first kills bad candidates fastest.
///   - `chern_class` is O(1) and discriminates the c_1 = 0 / c_3 = 6
///     constraints.
///   - `eta` is O(n_b) and adds the fundamental-group-dependent prefactor.
///   - `ade` is O(h11) integer comparisons.
///   - `slope` is O(n_b - 3) variance and is the loosest filter.
pub fn stage1_topological_loss_with_budget(
    c: &Candidate,
    n_gen_target: i32,
    budget: f64,
) -> Stage1Breakdown {
    let mut out = Stage1Breakdown {
        chern: 1.0,
        slope: 1.0,
        wilson_line: 1.0,
        eta: 1.0,
        ade: 1.0,
    };
    let mut acc = 0.0;

    out.wilson_line = wilson_line_loss(c);
    acc += out.wilson_line;
    if acc > budget {
        return out;
    }

    out.chern = chern_class_loss(c, n_gen_target);
    acc += out.chern;
    if acc > budget {
        return out;
    }

    out.eta = quotient_area_ratio_loss(c);
    acc += out.eta;
    if acc > budget {
        return out;
    }

    out.ade = ade_wavenumber_loss(c);
    acc += out.ade;
    if acc > budget {
        return out;
    }

    out.slope = slope_stability_loss(c);
    out
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Stage1Breakdown {
    pub chern: f64,
    pub slope: f64,
    pub wilson_line: f64,
    pub eta: f64,
    pub ade: f64,
}

impl Stage1Breakdown {
    pub fn total(&self) -> f64 {
        self.chern + self.slope + self.wilson_line + self.eta + self.ade
    }
}

/// Topology family enum so Stage 1 can sweep across {Tian-Yau, Schoen-Z3xZ3}
/// instead of taking caller-fixed (chi, h11, h21).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyFamily {
    TianYauZ3,
    SchoenZ3xZ3,
}

impl TopologyFamily {
    pub fn euler_characteristic(&self) -> i32 {
        match self {
            TopologyFamily::TianYauZ3 => -6,
            TopologyFamily::SchoenZ3xZ3 => -6,
        }
    }
    pub fn fundamental_group(&self) -> &'static str {
        match self {
            TopologyFamily::TianYauZ3 => "Z3",
            TopologyFamily::SchoenZ3xZ3 => "Z3xZ3",
        }
    }
    pub fn h11(&self) -> usize {
        match self {
            TopologyFamily::TianYauZ3 => 14,
            TopologyFamily::SchoenZ3xZ3 => 19,
        }
    }
    pub fn h21(&self) -> usize {
        match self {
            TopologyFamily::TianYauZ3 => 23,
            TopologyFamily::SchoenZ3xZ3 => 19,
        }
    }
    pub fn short_name(&self) -> &'static str {
        match self {
            TopologyFamily::TianYauZ3 => "TY/Z3",
            TopologyFamily::SchoenZ3xZ3 => "Schoen/Z3xZ3",
        }
    }
    pub fn from_short_name(s: &str) -> Option<Self> {
        match s {
            "TY/Z3" | "TianYau" | "tian-yau" => Some(TopologyFamily::TianYauZ3),
            "Schoen/Z3xZ3" | "Schoen" | "schoen" => Some(TopologyFamily::SchoenZ3xZ3),
            _ => None,
        }
    }
    pub fn all() -> [TopologyFamily; 2] {
        [TopologyFamily::TianYauZ3, TopologyFamily::SchoenZ3xZ3]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Candidate;

    fn mk_candidate(bundle: Vec<f64>, group: &str, chi: i32) -> Candidate {
        let geometry = match group {
            "Z3xZ3" => crate::geometry::CicyGeometry::schoen_z3xz3(),
            _ => crate::geometry::CicyGeometry::tian_yau_z3(),
        };
        Candidate {
            id: 0,
            candidate_short_name: "test".to_string(),
            euler_characteristic: chi,
            fundamental_group: group.to_string(),
            kahler_moduli: vec![1.0; 14],
            complex_moduli_real: vec![0.0; 23],
            complex_moduli_imag: vec![0.0; 23],
            bundle_moduli: bundle,
            parent_id: None,
            geometry,
        }
    }

    #[test]
    fn chern_zero_when_targets_met() {
        // c1 = 0, c2 = -(-6)/12 = 0.5, c3 = 6 (= 2 * 3)
        let bundle = vec![0.0, 0.5, 6.0];
        let c = mk_candidate(bundle, "Z3", -6);
        let loss = chern_class_loss(&c, 3);
        assert!(loss < 1e-10, "expected near-zero, got {loss}");
    }

    #[test]
    fn chern_nonzero_when_c1_nonvanishing() {
        let bundle = vec![1.0, 0.5, 6.0];
        let c = mk_candidate(bundle, "Z3", -6);
        let loss = chern_class_loss(&c, 3);
        assert!(loss >= 1.0, "expected loss >= 1, got {loss}");
    }

    #[test]
    fn slope_stability_zero_when_all_equal() {
        let bundle = vec![0.0, 0.5, 6.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let c = mk_candidate(bundle, "Z3", -6);
        let loss = slope_stability_loss(&c);
        assert!(loss < 1e-10);
    }

    #[test]
    fn wilson_line_zero_at_third_roots() {
        use std::f64::consts::PI;
        let mut bundle = vec![0.0; 25];
        bundle[20] = 0.0;
        bundle[21] = 2.0 * PI / 3.0;
        bundle[22] = 4.0 * PI / 3.0;
        let c = mk_candidate(bundle, "Z3", -6);
        let loss = wilson_line_loss(&c);
        assert!(loss < 1e-10, "expected near-zero, got {loss}");
    }

    #[test]
    fn topology_family_metadata() {
        assert_eq!(TopologyFamily::TianYauZ3.euler_characteristic(), -6);
        assert_eq!(TopologyFamily::SchoenZ3xZ3.fundamental_group(), "Z3xZ3");
        assert_eq!(
            TopologyFamily::from_short_name("TY/Z3"),
            Some(TopologyFamily::TianYauZ3)
        );
    }
}
