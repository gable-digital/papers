//! # Dynamic E_8 → E_6 × SU(3) sector decomposition for Yukawa modes
//!
//! ## What this fixes
//!
//! The legacy [`crate::yukawa_sectors`] module hard-codes the SU(5)
//! GUT split `[0,1,2] → up`, `[3,4,5] → down`, `[6,7,8] → leptons`
//! on a 9-mode harmonic spectrum. This is the right answer for an
//! `SU(5)` GUT *if* exactly 9 zero modes appear and *if* the modes
//! happen to be enumerated in that specific order. It is the wrong
//! answer for the actual heterotic E_8 → E_6 × SU(3) pattern, where
//! the 27 of E_6 decomposes under the SU(3) bundle structure group as
//!
//! ```text
//!     27  →  (3, 3̄)  ⊕  (3̄, 3)  ⊕  (1, 8)  ⊕  (1, 1)
//! ```
//!
//! For the standard-model embedding the relevant pieces are
//!
//! * **Up-quark sector** (10 of SU(5) inside 27 of E_6): from
//!   `H^1(M, V)` modes invariant under the Wilson line in the
//!   appropriate phase class.
//! * **Down-quark / lepton sector**: complementary phase class.
//! * **Higgs**: `H^1(M, V)` mode of the trivial Wilson-line phase
//!   (or `H^1(M, End V)` for the singlet Higgs).
//!
//! The actual sector assignment depends on which of the
//! `quotient_order` Wilson-line phase classes the mode lives in.
//! [`assign_sectors_dynamic`] performs this assignment by computing
//! the Wilson-line eigenvalue of each harmonic mode (via its
//! coefficient vector projected onto the line-bundle Cartan basis)
//! and grouping modes accordingly.
//!
//! ## Algorithm
//!
//! 1. Take the harmonic-mode coefficient vectors from
//!    [`crate::route34::zero_modes_harmonic`].
//! 2. For each mode `α`, compute its Wilson-line phase
//!    `e^{2π i ⟨w_E8, α(line)⟩ / N}` where `α(line)` is the
//!    line-bundle multidegree of the dominant component in the
//!    coefficient vector.
//! 3. Group modes by their phase class (3 classes for `Z/3`,
//!    9 classes for `Z/3 × Z/3`).
//! 4. The **up-quark** sector picks up phase 0 (real / invariant).
//!    The **down-quark / lepton** sectors share the non-trivial
//!    phase classes via the standard `Z/3` minimal-monad
//!    dictionary (DHOR 2006 Tab. 1).
//! 5. The Higgs is the lowest-eigenvalue (most-trivial-monomial)
//!    mode in the `End V` cohomology — for our monad-only setting
//!    we approximate it as the singlet of `H^1(V) ⊕ H^1(V*)`.
//!
//! Falls back gracefully to a deterministic equal-split when the
//! Wilson-line phase information is unavailable (e.g. trivial
//! quotient, in which case all modes share the same phase).
//!
//! ## References
//!
//! * Donagi, He, Ovrut, Reinbacher, "The particle spectrum of
//!   heterotic compactifications", JHEP **06** (2006) 039,
//!   arXiv:hep-th/0512149.
//! * Anderson, Lukas, Palti, "Two hundred heterotic standard
//!   models on smooth Calabi-Yau threefolds", arXiv:1106.4804 (2011).
//! * Slansky, R., "Group theory for unified model building",
//!   Phys. Rep. **79** (1981) 1.

use crate::route34::wilson_line_e8::WilsonLineE8;
use crate::route34::zero_modes_harmonic::HarmonicZeroModeResult;
use crate::zero_modes::MonadBundle;
use std::f64::consts::PI;

/// One sector's mode-index assignment: which harmonic-mode indices
/// (within the `HarmonicZeroModeResult.modes` Vec) belong to this
/// sector.
#[derive(Clone, Debug, Default)]
pub struct SectorAssignment {
    pub up_quark: Vec<usize>,
    pub down_quark: Vec<usize>,
    pub lepton: Vec<usize>,
    pub higgs: Vec<usize>,
}

/// Top-level dynamic sector assignment.
pub fn assign_sectors_dynamic(
    bundle: &MonadBundle,
    modes: &HarmonicZeroModeResult,
    wilson: &WilsonLineE8,
) -> SectorAssignment {
    let n_modes = modes.modes.len();
    if n_modes == 0 {
        return SectorAssignment::default();
    }

    let n = wilson.quotient_order.max(1) as i32;

    // Compute the Wilson-line phase class of each mode by examining
    // the dominant coefficient in its polynomial-seed expansion.
    // The line-bundle multidegree of seed i is `bundle.b_lines[i]`
    // (or, for 3-factor lifts, `bundle.b_lines_3factor[i]`).
    //
    // **P8.3-followup-B (this revision)** — the 2-factor projection
    // path (legacy below) computes
    //   inner = wilson.cartan_phases[6] * b[0]
    //         + wilson.cartan_phases[7] * b[1]
    // but the canonical Slansky-1981 Cartan phases are
    //   (2/3, -1/3, -1/3, 0, 0, 0, 0, 0)
    // so [6] and [7] are exactly zero — every 2-factor mode collapses
    // to phase class 0 and the round-robin fallback (line 183 below)
    // fires unconditionally. The 3-factor lift path uses the
    // Z/3×Z/3-aware projection
    //   class = (a − b) mod 3
    // on the 3-component bidegree `[a, b, c]`, which gives three
    // distinct classes for the canonical Schoen B-summand bidegrees
    // {[1,0,0], [0,1,0], [0,0,1]}.
    //
    // **Wilson-fix diagnostic (Apr 2026 — `p_wilson_fix_diagnostic.md`)**.
    // The legacy `cartan_phases[6] * b[0] + cartan_phases[7] * b[1]`
    // formulation was dimensionally inconsistent: `cartan_phases` is
    // an 8-vector in the orthonormal `R^8` Cartan realisation of E_8
    // (Bourbaki Plate VII), while `b` is a CP^N divisor degree on the
    // CY3 ambient — the two live in different spaces and pairing
    // their coordinates index-for-index is not a meaningful inner
    // product. The physically correct Wilson Z/3 character on AKLP-
    // style B-summands is the SU(3)-flavour position from the
    // splitting-principle decomposition `V = ⊕_a L_a`: each B-summand
    // index `a` carries a distinct SU(3)-Cartan weight, equivalently
    // (per `wilson_line_e8_z3xz3::fiber_character` Table at line 65)
    // `g_1 = a mod 3`. We use this per-summand-index projection on
    // the legacy 2-factor path, which gives three distinct phase
    // classes for the AKLP TY/Z3 bundle (B-summands 0..5 → classes
    // 0,1,2,0,1,2) instead of the all-zero collapse.
    //
    // Cross-reference:
    //   `references/p8_3_followup_a2_tensor_sparsity_diagnostic.md`
    //   `references/p_wilson_fix_diagnostic.md`
    //   `zero_modes::MonadBundle::schoen_z3xz3_canonical`
    //   `route34::wilson_line_e8_z3xz3::fiber_character`
    let mut phase_class: Vec<i32> = vec![0; n_modes];
    for (m_idx, m) in modes.modes.iter().enumerate() {
        // The harmonic-mode coefficient vector is in the EXPANDED
        // polynomial-seed basis (one entry per bigraded monomial, not
        // per B-summand). To classify the mode under the Wilson-line
        // Z/N action we need the dominant B-summand, not the dominant
        // monomial. Aggregate `|c_α|²` per B-line via the
        // `seed_to_b_line` map carried alongside the result.
        let mut per_b: Vec<f64> = vec![0.0; bundle.b_lines.len()];
        for (i, c) in m.coefficients.iter().enumerate() {
            let b_line = modes
                .seed_to_b_line
                .get(i)
                .copied()
                .unwrap_or(usize::MAX);
            if b_line < per_b.len() {
                per_b[b_line] += c.norm_sqr();
            }
        }
        let mut max_abs = 0.0f64;
        let mut dom = 0usize;
        for (i, p) in per_b.iter().enumerate() {
            if *p > max_abs {
                max_abs = *p;
                dom = i;
            }
        }
        if max_abs > 0.0 && dom < bundle.b_lines.len() {
            let class = if let Some(b3f) = bundle
                .b_lines_3factor
                .as_ref()
                .filter(|v| v.len() == bundle.b_lines.len())
            {
                // 3-factor Z/3×Z/3 path. Use `(a − b) mod 3` —
                // the SU(3)-Cartan projection of the canonical
                // Z/3×Z/3 Wilson generator on the first two ambient
                // factors. We always reduce mod 3 (NOT mod `n`)
                // because the by_class consumer below extracts
                // exactly three buckets `{0, 1, 2}` and maps them to
                // {lepton, up_quark, down_quark}; for `n = 9`
                // (Z/3×Z/3 quotient) the remaining 6 buckets `{3..8}`
                // would never feed any sector. Three distinct mod-3
                // classes for the canonical Schoen B-summand
                // bidegrees {[1,0,0], [0,1,0], [0,0,1]}.
                let b = b3f[dom];
                (b[0] - b[1]).rem_euclid(3)
            } else {
                // Legacy 2-factor path. The Wilson Z/N character on
                // AKLP-style B-summands is the SU(3)-flavour position
                // from the splitting-principle decomposition: each
                // B-summand index `a` carries SU(3)-Cartan weight
                // `a mod 3` (cf. `wilson_line_e8_z3xz3::fiber_character`
                // Table at line 65 — b_lines `[1,0]^3 ⊕ [0,1]^3` map
                // to characters `(0,1,2,0,1,2)`). We use the dominant
                // B-summand index `dom` reduced modulo `quotient_order`
                // (capped at 3, since the SU(3) flavour structure is
                // intrinsically Z/3 — for `n = 9` Z/3×Z/3 quotients
                // the second character requires the 3-factor branch
                // above). This is the physically correct projection;
                // the prior `cartan_phases[6]*b[0] + cartan_phases[7]*b[1]`
                // formulation was dimensionally inconsistent (paired
                // an E_8 Cartan component with a CY3 divisor degree)
                // and always returned zero because `cartan_phases[6]`
                // and `cartan_phases[7]` are zero in the canonical
                // E_8 → E_6 × SU(3) embedding — see
                // `references/p_wilson_fix_diagnostic.md`.
                //
                // We tap `wilson.cartan_phases` to keep the binding
                // live (so a future re-indexing of cartan_phases
                // does not silently change behaviour here): the sum
                // is identically zero in the canonical convention
                // (the SU(3) Cartan triple `(2/3, -1/3, -1/3)` sums
                // to 0), but the call documents the dependency on
                // the canonical Wilson element.
                let _trace = wilson.cartan_phases[0]
                    + wilson.cartan_phases[1]
                    + wilson.cartan_phases[2];
                debug_assert!(_trace.abs() < 1.0e-12,
                    "canonical SU(3)-Cartan must be traceless");
                (dom as i32).rem_euclid(3.min(n).max(1))
            };
            phase_class[m_idx] = class.rem_euclid(n);
        }
    }

    let _ = PI;

    // Group modes by phase class.
    let mut by_class: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
    for (idx, c) in phase_class.iter().enumerate() {
        let bin = ((*c as usize) % (n as usize)).min(by_class.len() - 1);
        by_class[bin].push(idx);
    }

    // DHOR 2006 / ALP 2011 minimal-monad dictionary (Z/3 case):
    //   class 0 → Higgs / singlet sector;
    //   class 1 → up-quark sector (and lepton-doublet under SU(5));
    //   class 2 → down-quark / charged-lepton-singlet sector.
    //
    // We map onto the canonical assignment with graceful fallbacks:
    //   - if any sector is empty, fall back to round-robin assignment
    //     of the remaining unassigned modes.
    let class_count = by_class.len();
    let mut up: Vec<usize> = if class_count > 1 {
        by_class[1].clone()
    } else {
        Vec::new()
    };
    let mut down: Vec<usize> = if class_count > 2 {
        by_class[2].clone()
    } else {
        Vec::new()
    };
    let mut lepton: Vec<usize> = if class_count > 0 {
        by_class[0].clone()
    } else {
        Vec::new()
    };
    let mut higgs: Vec<usize> = Vec::new();

    // Round-robin equal-split fallback if any sector ended up empty
    // (the `Z/1` trivial case, or modes that all collapsed to the
    // same phase class because of finite-precision wilson_line_e8
    // phases).
    if up.is_empty() || down.is_empty() || lepton.is_empty() {
        up.clear();
        down.clear();
        lepton.clear();
        for i in 0..n_modes {
            match i % 3 {
                0 => up.push(i),
                1 => down.push(i),
                _ => lepton.push(i),
            }
        }
    }

    // Higgs sector population. The journal §F.1 tree-level Higgs
    // prescription identifies the Standard-Model Higgs doublet with
    // the **lowest-eigenvalue** (h_0) harmonic mode of the trivial-
    // character (phase class 0) sub-bundle: only that zero-mode
    // carries the 246 GeV electroweak vev at tree level, while every
    // higher harmonic excitation `h_n` (n ≥ 1) is a massive scalar
    // with `⟨h_n⟩ = 0` and contributes nothing to the EW-scale
    // fermion mass matrix.
    //
    // P8.3-followup-C (this revision): we still populate `higgs`
    // with the **full** list of phase-class-0 candidate-mode indices
    // for diagnostic / downstream use, but we **sort the list by
    // ascending harmonic eigenvalue** so that `higgs[0]` is always
    // the lowest-eigenvalue (h_0) mode. The downstream contraction
    // [`extract_3x3_from_tensor`] then uses only `higgs[0]`. This
    // supersedes the earlier P8.3b uniform-sum contraction
    // `Y_{ij} = Σ_h v_h T_{i,j,h}`, which mixed massive scalar tower
    // excitations into the EW mass matrix and was physically
    // incorrect (P8.3c hostile-review verdict).
    //
    // Population paths, in order of preference:
    //
    //   1. Wilson-line phase-class-0 modes (the journal-prescribed
    //      path), when those classes separate cleanly.
    //   2. Round-robin fallback: with the trivial-quotient or
    //      collapsed-phase-class case, populate `higgs` with every
    //      third mode starting from index 0 so the lowest-eigenvalue
    //      pick still has multiple candidates to choose from.
    //   3. Final defensive fallback: all modes.
    //
    // After population, `higgs` is sorted by ascending
    // `modes.modes[i].eigenvalue` so `higgs[0]` is the lowest-
    // eigenvalue Higgs zero-mode (h_0).
    let class_count_for_higgs = by_class.len();
    if class_count_for_higgs > 0 && !by_class[0].is_empty() {
        higgs = by_class[0].clone();
    } else {
        for i in (0..n_modes).step_by(3) {
            higgs.push(i);
        }
        if higgs.is_empty() {
            higgs.extend(0..n_modes);
        }
    }
    // Sort `higgs` by ascending harmonic eigenvalue so `higgs[0]`
    // is the lowest-eigenvalue (h_0) Higgs zero-mode per journal
    // §F.1. Non-finite eigenvalues sort to the end. This is the
    // identification path for the tree-level EW vev carrier.
    higgs.sort_by(|&a, &b| {
        let ea = modes
            .modes
            .get(a)
            .map(|m| m.eigenvalue)
            .unwrap_or(f64::INFINITY);
        let eb = modes
            .modes
            .get(b)
            .map(|m| m.eigenvalue)
            .unwrap_or(f64::INFINITY);
        let ea = if ea.is_finite() { ea } else { f64::INFINITY };
        let eb = if eb.is_finite() { eb } else { f64::INFINITY };
        ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
    });

    SectorAssignment {
        up_quark: up,
        down_quark: down,
        lepton,
        higgs,
    }
}

/// Extract a 3×3 (re, im) Yukawa matrix from a
/// [`crate::route34::yukawa_overlap_real::Tensor3`] triplet
/// `(left_indices, right_indices, higgs_indices)`. Returns the matrix
/// in the same `[[(f64, f64); 3]; 3]` shape as
/// [`crate::yukawa_overlap::YukawaSpectrum`] for downstream PDG / RG
/// consumption.
///
/// **Higgs-vev contraction — tree-level h_0 only (P8.3-followup-C).**
/// The Yukawa coupling tensor `T_{ijk}` is over harmonic modes; the
/// physical 3×3 fermion mass matrix is the contraction with the
/// Higgs vev,
///
/// ```text
///     Y_{ij}  =  v_h0  T_{i, j, h_0}
/// ```
///
/// where `h_0` is the **lowest-harmonic-eigenvalue** Higgs zero-mode
/// (the smallest `eigenvalue` field of any
/// [`crate::route34::zero_modes_harmonic::HarmonicMode`] in the
/// trivial-character / phase-class-0 sub-bundle). Per the journal
/// §F.1 tree-level Higgs prescription, only this zero-mode carries
/// the 246 GeV electroweak vev at tree level — every higher harmonic
/// excitation `h_n` (n ≥ 1) is a massive scalar with `⟨h_n⟩ = 0` and
/// must NOT contribute to the EW-scale fermion mass matrix.
///
/// **Why not the uniform sum.** The earlier P8.3b revision contracted
/// uniformly across ALL phase-class-0 modes,
/// `Y_{ij} = Σ_h v_h T_{i,j,h}` with `v_h ≡ 1`. The P8.3c hostile
/// review flagged this as physically incorrect: it mixes the massive
/// harmonic-tower scalars (which have zero vev) into the EW mass
/// matrix, effectively assigning each massive scalar a 246 GeV
/// vev. The correct tree-level prescription uses only `h_0`.
///
/// **Caller contract.** [`assign_sectors_dynamic`] now sorts
/// `sectors.higgs` by ascending harmonic eigenvalue, so
/// `higgs_indices[0]` is the lowest-eigenvalue Higgs zero-mode.
/// We use only that single index. Trailing entries are intentionally
/// ignored at this layer (they remain available for diagnostic /
/// future radiative-correction extensions, where massive harmonic
/// excitations enter through 1-loop graphs rather than tree-level
/// vevs).
///
/// `left_indices` and `right_indices` are 3-mode lists picked by
/// [`assign_sectors_dynamic`] for the left- and right-handed
/// fermions. Only the first 3 entries of each are used.
///
/// **Under-sized sectors are zero-padded (P8.3-followup-B2).** When
/// a class bucket holds fewer than 3 modes (e.g. the synthetic
/// Schoen lepton class with only 2 class-0 modes), the missing rows
/// or columns of `Y` are left as zero — preserving the empirical
/// rank of the class. The earlier duplicate-pad behaviour
/// (`unwrap_or(*v.last())`) collapsed genuine partial-rank matrices
/// onto a single corner and is no longer used.
///
/// `higgs_indices` is the (sorted) list of harmonic-mode indices
/// identified as Higgs candidates by [`assign_sectors_dynamic`].
/// **Only the first entry (`higgs_indices[0]` = h_0, the lowest
/// harmonic eigenvalue) is used in the contraction.** If empty,
/// returns the zero matrix (defensive — caller must guarantee a
/// non-empty Higgs sector for a physical prediction).
///
/// **Note on P8.3-followup-A.** The upstream
/// `solve_harmonic_zero_modes` currently produces a near-rank-1
/// `T_{ijk}` because the harmonic-mode polynomial-seed expansion
/// coalesces under Galerkin orthogonalisation. With a rank-1 `T`
/// the single-h_0 contraction `Y_{ij} = T_{i,j,h_0}` will still
/// be near-rank-1 — but this is now the upstream issue's
/// signature, not a contraction-layer artifact. Once
/// P8.3-followup-A produces functionally distinct harmonic modes,
/// this contraction becomes the correct tree-level fermion mass
/// matrix.
pub fn extract_3x3_from_tensor(
    couplings: &crate::route34::yukawa_overlap_real::Tensor3,
    left_indices: &[usize],
    right_indices: &[usize],
    higgs_indices: &[usize],
) -> [[(f64, f64); 3]; 3] {
    let mut m: [[(f64, f64); 3]; 3] = [[(0.0, 0.0); 3]; 3];
    let n = couplings.n;
    if n == 0 {
        return m;
    }
    // **P8.3-followup-B2 fix.** Honest zero-pad for under-sized
    // sectors. When a class bucket has fewer than 3 modes (e.g. the
    // synthetic Schoen lepton bucket with only 2 class-0 modes), the
    // earlier duplicate-pad behaviour `unwrap_or(*v.last())` caused
    // the same `(li, rj, h0)` tensor entry to be written into two
    // different Y[i,j] cells — collapsing the genuine partial-rank
    // matrix into an artificial rank-1 block on a single corner.
    // This regressed Y_e from 5/9 to 1/9 bucket-hits in the P8.3-
    // followup-A2 sparsity diagnostic.
    //
    // Replacement contract: `lookup` returns `Some(idx)` when the
    // requested index is in range, `None` otherwise. The contraction
    // loop below leaves `Y[g_left, g_right] = 0` for any out-of-range
    // entry, preserving the empirical rank of the class (rank-2 for
    // a 2-mode bucket, rank-1 for a 1-mode bucket, rank-0 if empty).
    // Cross-reference: references/p8_3_followup_a2_tensor_sparsity_diagnostic.md
    let lookup = |v: &[usize], k: usize| -> Option<usize> {
        v.get(k).copied()
    };

    // Pick the lowest-harmonic-eigenvalue Higgs index (h_0). Caller
    // ([`assign_sectors_dynamic`]) sorts `higgs_indices` ascending
    // by eigenvalue, so `higgs_indices[0]` is h_0. Empty list →
    // return the zero matrix; caller is responsible for guaranteeing
    // a populated Higgs sector for a physical prediction (see
    // journal §F.1).
    let h0 = match higgs_indices.iter().copied().find(|&h| h < n) {
        Some(h) => h,
        None => return m,
    };

    // Tree-level single-doublet vev coefficient. The downstream
    // `run_yukawas_to_mz` rescales by `v_higgs = 246 GeV`, so the
    // absolute scalar magnitude lives outside this contraction.
    let v_h0: f64 = 1.0;

    for i in 0..3 {
        for j in 0..3 {
            let li = match lookup(left_indices, i) {
                Some(idx) if idx < n => idx,
                _ => {
                    // Zero-pad: under-sized left sector or
                    // out-of-bounds index. Y[i, j] stays (0, 0).
                    continue;
                }
            };
            let rj = match lookup(right_indices, j) {
                Some(idx) if idx < n => idx,
                _ => {
                    // Zero-pad: under-sized right sector or
                    // out-of-bounds index. Y[i, j] stays (0, 0).
                    continue;
                }
            };
            let z = couplings.entry(li, rj, h0);
            let (re, im) = if z.re.is_finite() && z.im.is_finite() {
                (v_h0 * z.re, v_h0 * z.im)
            } else {
                (0.0, 0.0)
            };
            m[i][j] = (re, im);
        }
    }
    m
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route34::hym_hermitian::{
        solve_hym_metric, HymConfig, InMemoryMetricBackground,
    };
    use crate::route34::zero_modes_harmonic::{
        solve_harmonic_zero_modes, HarmonicConfig,
    };
    use crate::zero_modes::AmbientCY3;
    use num_complex::Complex64;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    fn synthetic_metric(n_pts: usize, seed: u64) -> InMemoryMetricBackground {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut points = Vec::with_capacity(n_pts);
        for _ in 0..n_pts {
            let mut p = [Complex64::new(0.0, 0.0); 8];
            for k in 0..8 {
                let re: f64 = rng.random_range(-1.0..1.0);
                let im: f64 = rng.random_range(-1.0..1.0);
                p[k] = Complex64::new(re, im);
            }
            points.push(p);
        }
        let w_each = 1.0 / (n_pts as f64);
        InMemoryMetricBackground {
            points,
            weights: vec![w_each; n_pts],
            omega: vec![Complex64::new(1.0, 0.0); n_pts],
        }
    }

    /// Test: dynamic sectorisation produces non-empty sectors on the
    /// ALP example.
    #[test]
    fn dynamic_sectors_nonempty() {
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let ambient = AmbientCY3::tian_yau_upstairs();
        let metric = synthetic_metric(60, 41);
        let h_v = solve_hym_metric(&bundle, &metric, &HymConfig::default());
        let modes = solve_harmonic_zero_modes(&bundle, &ambient, &metric, &h_v, &HarmonicConfig::default());
        if modes.modes.is_empty() {
            return;
        }
        let wilson = WilsonLineE8::canonical_e8_to_e6_su3(3);
        let s = assign_sectors_dynamic(&bundle, &modes, &wilson);
        // At least one sector should have at least one mode.
        let total = s.up_quark.len() + s.down_quark.len() + s.lepton.len();
        assert!(total >= 1, "all sectors empty");
    }

    /// Test: extract_3x3_from_tensor returns finite values on a
    /// non-trivial `Tensor3`.
    #[test]
    fn extract_3x3_returns_finite() {
        let mut t = crate::route34::yukawa_overlap_real::Tensor3::zeros(4);
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    t.set(i, j, k, Complex64::new((i + j + k) as f64, 0.0));
                }
            }
        }
        let m = extract_3x3_from_tensor(&t, &[0, 1, 2], &[1, 2, 3], &[0]);
        for i in 0..3 {
            for j in 0..3 {
                assert!(m[i][j].0.is_finite() && m[i][j].1.is_finite());
            }
        }
        // (i=0, j=0) entry should be t[0, 1, 0] = 1 (single-Higgs
        // contraction with higgs_indices=[0] and v_h=1).
        assert!((m[0][0].0 - 1.0).abs() < 1e-12);
    }

    /// **Rank-3 Yukawa regression test (P8.3-followup-C).** A
    /// synthetic full-rank Yukawa tensor `T_{ijk}` whose `h_0`
    /// slice `T[:,:,h_0]` is generically full-rank should yield a
    /// rank-3 3×3 mass matrix under the lowest-harmonic-eigenvalue
    /// (h_0) contraction. The pre-P8.3b single-`h_0` contraction
    /// produced rank-1 only when `T_{ijk}` itself was rank-1 along
    /// the third index — which is the upstream P8.3-followup-A
    /// issue, not a contraction-layer bug. With a generic random
    /// `T` whose third-index slices are independent matrices, the
    /// h_0 slice is rank-3 and the contraction recovers it.
    #[test]
    fn extract_3x3_yukawa_returns_rank_3_on_synthetic_full_rank_tensor() {
        let n = 5;
        let mut t = crate::route34::yukawa_overlap_real::Tensor3::zeros(n);
        // Strongly non-factorisable deterministic tensor: each
        // (i,j,k) entry is hashed through a coprime-index scheme
        // that breaks any rank-1 outer-product factorisation along
        // the third index. The `h_0` (k=2) slice is therefore a
        // generic 5×5 random complex matrix, which is rank-5
        // (and hence rank-3 on the (0..3)×(0..3) sub-block) with
        // probability 1.
        let mut state: u64 = 0xCAFE_F00D_BEEF_DEAD;
        let mut next = || -> f64 {
            state = state.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let z = Complex64::new(next(), next());
                    t.set(i, j, k, z);
                }
            }
        }
        let left = vec![0usize, 1, 2];
        let right = vec![0usize, 1, 2];
        // P8.3-followup-C: the contraction now uses ONLY higgs[0]
        // (the lowest-harmonic-eigenvalue h_0). Caller
        // (`assign_sectors_dynamic`) is responsible for sorting the
        // list ascending by eigenvalue; here we pass [2, 3, 4] with
        // 2 as h_0.
        let higgs = vec![2usize, 3, 4];

        let m = extract_3x3_from_tensor(&t, &left, &right, &higgs);

        // Repack the (re, im) pairs into a complex 3x3 matrix and
        // verify it is full-rank by computing the determinant.
        let m_c: [[Complex64; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| Complex64::new(m[i][j].0, m[i][j].1))
        });
        let det = m_c[0][0] * (m_c[1][1] * m_c[2][2] - m_c[1][2] * m_c[2][1])
            - m_c[0][1] * (m_c[1][0] * m_c[2][2] - m_c[1][2] * m_c[2][0])
            + m_c[0][2] * (m_c[1][0] * m_c[2][1] - m_c[1][1] * m_c[2][0]);
        let det_mag = det.norm();
        assert!(
            det_mag > 1.0e-6,
            "Yukawa matrix has near-zero determinant (rank-1 collapse): det = {:e}",
            det_mag
        );
        // All three rows nonzero (no row collapse).
        for i in 0..3 {
            let row_norm: f64 = (0..3)
                .map(|j| m[i][j].0 * m[i][j].0 + m[i][j].1 * m[i][j].1)
                .sum::<f64>()
                .sqrt();
            assert!(
                row_norm > 1.0e-6,
                "Yukawa matrix row {} collapsed: norm = {:e}",
                i,
                row_norm
            );
        }
    }

    /// **Empty-higgs defensive contraction (P8.3-followup-C).**
    /// With an empty `higgs_indices` slice the function returns
    /// the zero matrix. The pre-P8.3-followup-C uniform-sum-over-
    /// all-modes fallback was physically incorrect (mixed massive
    /// scalar tower into the EW mass matrix) and has been removed;
    /// callers must guarantee a populated Higgs sector for a
    /// physical prediction.
    #[test]
    fn extract_3x3_empty_higgs_returns_zero_matrix() {
        let mut t = crate::route34::yukawa_overlap_real::Tensor3::zeros(4);
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    t.set(i, j, k, Complex64::new((i + 2 * j + 3 * k) as f64, 0.0));
                }
            }
        }
        let m = extract_3x3_from_tensor(&t, &[0, 1, 2], &[0, 1, 2], &[]);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    m[i][j],
                    (0.0, 0.0),
                    "empty Higgs sector must yield zero matrix; got nonzero at ({},{}): {:?}",
                    i,
                    j,
                    m[i][j]
                );
            }
        }
    }

    /// **P8.3-followup-C regression test.** The contraction must use
    /// **only** `higgs_indices[0]` (the lowest-harmonic-eigenvalue
    /// h_0, per journal §F.1 tree-level Higgs prescription) and
    /// must NOT sum across the harmonic tower. Construct a tensor
    /// where `T[i, j, h_0]` and `T[i, j, h_1]` differ; verify
    /// `Y[i, j]` matches `T[i, j, h_0]` exactly.
    #[test]
    fn extract_3x3_uses_lowest_harmonic_higgs_only() {
        let n = 4;
        let mut t = crate::route34::yukawa_overlap_real::Tensor3::zeros(n);
        // Construct a tensor where the third-index slices differ
        // by index-dependent multiplicative factor:
        //   T[i, j, k] = (1 + i + 2j) * (k + 1) * (1 + i)
        // Crucially, T[:, :, h_0=1] ≠ T[:, :, h_1=2].
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let v = ((1 + i + 2 * j) as f64)
                        * ((k + 1) as f64)
                        * ((1 + i) as f64);
                    t.set(i, j, k, Complex64::new(v, 0.0));
                }
            }
        }
        // Caller-sorted: h_0 = 1 (lowest), h_1 = 2 (next), h_2 = 3.
        let higgs = vec![1usize, 2, 3];
        let m = extract_3x3_from_tensor(&t, &[0, 1, 2], &[0, 1, 2], &higgs);

        // Check Y[i,j] == T[i, j, h_0=1] (NOT the sum across {1,2,3}).
        for i in 0..3 {
            for j in 0..3 {
                let expected = ((1 + i + 2 * j) as f64) * 2.0 * ((1 + i) as f64);
                assert!(
                    (m[i][j].0 - expected).abs() < 1e-9,
                    "Y[{}, {}] must equal T[{}, {}, h_0=1] = {} (NOT sum); got {}",
                    i,
                    j,
                    i,
                    j,
                    expected,
                    m[i][j].0
                );
                // Sanity: NOT the sum over {1,2,3}, which would be
                //   (1+i+2j) * (1+i) * (2+3+4) = (1+i+2j)(1+i)*9.
                let uniform_sum =
                    ((1 + i + 2 * j) as f64) * ((1 + i) as f64) * 9.0;
                if (uniform_sum - expected).abs() > 1e-6 {
                    assert!(
                        (m[i][j].0 - uniform_sum).abs() > 1e-6,
                        "regression: Y[{}, {}] picked up the uniform-sum value {}",
                        i,
                        j,
                        uniform_sum
                    );
                }
            }
        }
    }

    /// **P8.3-followup-C regression test.** `assign_sectors_dynamic`
    /// must sort `sectors.higgs` ascending by harmonic eigenvalue,
    /// so the downstream contraction sees `h_0` (lowest-eigenvalue
    /// mode) at index 0.
    #[test]
    fn assign_sectors_dynamic_sorts_higgs_by_eigenvalue() {
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let ambient = AmbientCY3::tian_yau_upstairs();
        let metric = synthetic_metric(60, 4242);
        let h_v = solve_hym_metric(&bundle, &metric, &HymConfig::default());
        let modes = solve_harmonic_zero_modes(
            &bundle,
            &ambient,
            &metric,
            &h_v,
            &HarmonicConfig::default(),
        );
        if modes.modes.len() < 2 {
            // Documented bail: synthetic clouds may collapse the kernel
            // basis. Test cannot exercise the sort path with < 2 modes.
            return;
        }
        let wilson = WilsonLineE8::canonical_e8_to_e6_su3(3);
        let sectors = assign_sectors_dynamic(&bundle, &modes, &wilson);
        if sectors.higgs.len() < 2 {
            return;
        }
        // Verify ascending eigenvalue order.
        for w in sectors.higgs.windows(2) {
            let ea = modes.modes[w[0]].eigenvalue;
            let eb = modes.modes[w[1]].eigenvalue;
            assert!(
                ea <= eb || !ea.is_finite() || !eb.is_finite(),
                "sectors.higgs must be sorted ascending by eigenvalue; \
                 got {} -> {} at indices {} -> {}",
                ea,
                eb,
                w[0],
                w[1]
            );
        }
    }

    /// **P8.3-followup-B2 regression test.** When a sector class has
    /// fewer than 3 modes, `extract_3x3_from_tensor` MUST zero-pad the
    /// missing rows/columns instead of duplicate-padding with the last
    /// available index. The earlier `unwrap_or(*v.last())` pad caused
    /// the synthetic Schoen lepton bucket (2 modes, class-0) to land
    /// `T[1, 1, 0]` (and other duplicates) onto multiple Y[i,j] cells
    /// — collapsing a genuine rank-2 partial matrix into a rank-1
    /// block on a single corner. The regressed Y_e bucket-hit dropped
    /// from 5/9 (pre-3-factor bundle) to 1/9 (post-3-factor with
    /// duplicate-pad bug); after this fix the honest report is 2/9
    /// (the 2 genuine class-0 modes, no duplication).
    ///
    /// Cross-reference:
    ///   `references/p8_3_followup_a2_tensor_sparsity_diagnostic.md`
    #[test]
    fn extract_3x3_zero_pads_undersized_sector() {
        // 9x9x9 distinct-entry tensor: T[i,j,k] = (i+1) + 0.1*(j+1) + 0.01*(k+1).
        let n = 9;
        let mut t = crate::route34::yukawa_overlap_real::Tensor3::zeros(n);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let v = ((i + 1) as f64)
                        + 0.1 * ((j + 1) as f64)
                        + 0.01 * ((k + 1) as f64);
                    t.set(i, j, k, Complex64::new(v, 0.0));
                }
            }
        }
        // Regression case: 2 left modes (class-0 lepton bucket),
        // 3 right modes, 1 Higgs.
        let left = vec![0usize, 1];
        let right = vec![0usize, 1, 2];
        let higgs = vec![0usize];

        let m = extract_3x3_from_tensor(&t, &left, &right, &higgs);

        // Expected populated entries (i in {0, 1}, j in {0, 1, 2}):
        //   Y[i, j] = T[i, j, 0] = (i+1) + 0.1*(j+1) + 0.01
        for i in 0..2usize {
            for j in 0..3usize {
                let expected = ((i + 1) as f64)
                    + 0.1 * ((j + 1) as f64)
                    + 0.01;
                assert!(
                    (m[i][j].0 - expected).abs() < 1e-12,
                    "Y[{}, {}] expected {} (= T[{}, {}, 0]); got {}",
                    i,
                    j,
                    expected,
                    i,
                    j,
                    m[i][j].0
                );
                assert!(
                    m[i][j].1.abs() < 1e-12,
                    "Y[{}, {}].im expected 0; got {}",
                    i,
                    j,
                    m[i][j].1
                );
            }
        }
        // Zero-pad row (i = 2): every column must be (0, 0).
        for j in 0..3usize {
            assert_eq!(
                m[2][j],
                (0.0, 0.0),
                "Y[2, {}] must be zero-padded (under-sized left sector); \
                 got {:?} — duplicate-pad regression",
                j,
                m[2][j]
            );
        }
        // Cross-check: explicit duplicate-pad regression signature.
        // The buggy pad would have written T[1, j, 0] into Y[2, j],
        // i.e. (2 + 0.1*(j+1) + 0.01). Verify we did NOT.
        for j in 0..3usize {
            let dup_pad_value = 2.0 + 0.1 * ((j + 1) as f64) + 0.01;
            assert!(
                (m[2][j].0 - dup_pad_value).abs() > 1e-9,
                "Y[2, {}] picked up duplicate-pad value {} (regression)",
                j,
                dup_pad_value
            );
        }
    }

    /// **P8.3-followup-B regression test.** With the new 3-factor
    /// Schoen bundle, the round-robin fallback at line 183 of
    /// `assign_sectors_dynamic` MUST disengage — all three canonical
    /// class buckets (up_quark = class 1, down_quark = class 2,
    /// lepton = class 0) must be non-empty, and every mode must land
    /// in exactly one of them by phase-class projection (NOT by
    /// `i % 3` round-robin).
    ///
    /// Construction: synthesize a 6-mode `HarmonicZeroModeResult`
    /// whose `seed_to_b_line` map binds each mode 1:1 to a B-summand
    /// of `MonadBundle::schoen_z3xz3_canonical`. The 6 B-summands
    /// have 3-factor bidegrees
    ///   {[1,0,0]×2, [0,1,0]×2, [0,0,1]×2}
    /// projecting to phase classes `(a − b) mod 3`:
    ///   {1, 1, 2, 2, 0, 0}
    /// — three distinct classes, each with two modes.
    ///
    /// Expected: `up_quark = [0, 1]`, `down_quark = [2, 3]`,
    /// `lepton = [4, 5]`. NO round-robin scrambling.
    #[test]
    fn assign_sectors_dynamic_no_fallback_under_3factor_bundle() {
        use crate::route34::zero_modes_harmonic::{
            HarmonicMode, HarmonicZeroModeResult,
        };

        let bundle = MonadBundle::schoen_z3xz3_canonical();
        assert!(
            bundle.b_lines_3factor.is_some(),
            "schoen_z3xz3_canonical must populate b_lines_3factor"
        );
        assert_eq!(
            bundle.b_lines.len(),
            6,
            "expected 6 B-summands in the canonical Schoen bundle"
        );

        // Build a synthetic harmonic-mode result: one mode per
        // B-summand, dominant on its own line. Use seed_to_b_line to
        // bind each coefficient to the corresponding B-line.
        let n_b = bundle.b_lines.len();
        let mut modes = Vec::with_capacity(n_b);
        for i in 0..n_b {
            let mut coeffs = vec![Complex64::new(0.0, 0.0); n_b];
            coeffs[i] = Complex64::new(1.0, 0.0);
            modes.push(HarmonicMode {
                values: vec![],
                coefficients: coeffs,
                residual_norm: 0.0,
                eigenvalue: i as f64, // distinct, ascending
            });
        }
        let seed_to_b_line: Vec<usize> = (0..n_b).collect();
        let result = HarmonicZeroModeResult {
            modes,
            residual_norms: vec![0.0; n_b],
            seed_to_b_line,
            seed_basis_dim: n_b,
            cohomology_dim_predicted: n_b,
            cohomology_dim_observed: n_b,
            ..HarmonicZeroModeResult::default()
        };

        // Use Z/3 quotient (the projection (a − b) mod N reduces mod
        // 3 for the canonical Schoen B-summands).
        let wilson = WilsonLineE8::canonical_e8_to_e6_su3(3);
        let sectors = assign_sectors_dynamic(&bundle, &result, &wilson);

        // Verify three non-empty class buckets.
        assert!(!sectors.up_quark.is_empty(), "up_quark empty (class 1)");
        assert!(!sectors.down_quark.is_empty(), "down_quark empty (class 2)");
        assert!(!sectors.lepton.is_empty(), "lepton empty (class 0)");

        // Phase-class projection (a − b) mod 3 on the 3-factor
        // bidegrees:
        //   [1,0,0] -> class 1  (up_quark)
        //   [0,1,0] -> class 2  (down_quark)
        //   [0,0,1] -> class 0  (lepton)
        // Modes 0,1 dominant on b_line 0,1 -> class 1 -> up_quark.
        // Modes 2,3 dominant on b_line 2,3 -> class 2 -> down_quark.
        // Modes 4,5 dominant on b_line 4,5 -> class 0 -> lepton.
        let mut up_sorted = sectors.up_quark.clone();
        up_sorted.sort_unstable();
        let mut down_sorted = sectors.down_quark.clone();
        down_sorted.sort_unstable();
        let mut lep_sorted = sectors.lepton.clone();
        lep_sorted.sort_unstable();
        assert_eq!(up_sorted, vec![0, 1], "up_quark mode list");
        assert_eq!(down_sorted, vec![2, 3], "down_quark mode list");
        assert_eq!(lep_sorted, vec![4, 5], "lepton mode list");

        // Cross-check: round-robin would have given
        //   up = [0, 3], down = [1, 4], lepton = [2, 5]
        // — a clear scramble across phase classes. Verify we did NOT
        // get that pattern.
        assert_ne!(up_sorted, vec![0, 3], "round-robin fallback fired (bug)");
    }

    /// **Wilson-fix regression test (Apr 2026 —
    /// `references/p_wilson_fix_diagnostic.md`).** The legacy 2-factor
    /// projection in [`assign_sectors_dynamic`] previously computed
    /// the phase class via
    ///   `inner = wilson.cartan_phases[6]*b[0]
    ///         + wilson.cartan_phases[7]*b[1]`
    /// against the canonical E_8 → E_6 × SU(3) element
    ///   `cartan_phases = (2/3, -1/3, -1/3, 0, 0, 0, 0, 0)`
    /// — components [6] and [7] are exactly zero, so every mode
    /// collapsed to phase class 0 regardless of its B-summand
    /// bidegree. This regressed Y_d to 6/9 and Y_e to 4/9 bucket-hits
    /// for the TY/Z3 path (see
    /// `references/p8_3_followup_a2_tensor_sparsity_diagnostic.md`).
    ///
    /// Fix: project on the dominant B-summand index modulo 3 (the
    /// SU(3) flavour position from the splitting principle, matching
    /// `wilson_line_e8_z3xz3::fiber_character`'s convention).
    ///
    /// Test: build a 2-factor bundle (AKLP-style with 6 B-summands
    /// and `b_lines_3factor = None`) and synthetic 6-mode harmonic
    /// result, each mode 1:1 on its own B-summand. Assert the 2-factor
    /// projection produces ≥ 2 distinct phase classes (not the
    /// all-zero collapse).
    #[test]
    fn assign_sectors_dynamic_2factor_projection_distinguishes_classes() {
        use crate::route34::zero_modes_harmonic::{
            HarmonicMode, HarmonicZeroModeResult,
        };

        // AKLP TY/Z3 example: 2-factor bundle, b_lines_3factor = None.
        let bundle = MonadBundle::anderson_lukas_palti_example();
        assert!(
            bundle.b_lines_3factor.is_none(),
            "AKLP example must NOT have a 3-factor lift (this test \
             targets the legacy 2-factor projection branch)"
        );
        assert_eq!(
            bundle.b_lines.len(),
            6,
            "expected 6 B-summands in the AKLP example"
        );

        // Build a synthetic harmonic-mode result: one mode per
        // B-summand, dominant on its own line.
        let n_b = bundle.b_lines.len();
        let mut modes = Vec::with_capacity(n_b);
        for i in 0..n_b {
            let mut coeffs = vec![Complex64::new(0.0, 0.0); n_b];
            coeffs[i] = Complex64::new(1.0, 0.0);
            modes.push(HarmonicMode {
                values: vec![],
                coefficients: coeffs,
                residual_norm: 0.0,
                eigenvalue: i as f64,
            });
        }
        let seed_to_b_line: Vec<usize> = (0..n_b).collect();
        let result = HarmonicZeroModeResult {
            modes,
            residual_norms: vec![0.0; n_b],
            seed_to_b_line,
            seed_basis_dim: n_b,
            cohomology_dim_predicted: n_b,
            cohomology_dim_observed: n_b,
            ..HarmonicZeroModeResult::default()
        };

        let wilson = WilsonLineE8::canonical_e8_to_e6_su3(3);
        let sectors = assign_sectors_dynamic(&bundle, &result, &wilson);

        // The Wilson-fix projects on the dominant B-summand index mod
        // 3, so the 6 modes (one per B-summand) split as
        //   indices 0,3 -> class 0 (lepton)
        //   indices 1,4 -> class 1 (up_quark)
        //   indices 2,5 -> class 2 (down_quark)
        // — three distinct, non-empty buckets. The ROUND-ROBIN
        // FALLBACK MUST NOT FIRE; if the legacy all-zero projection
        // were still in place all 6 modes would land in class 0,
        // up_quark and down_quark would be empty, and the round-robin
        // would scramble them into up=[0,3], down=[1,4], lepton=[2,5].
        // The Wilson-fix happens to give the SAME index lists for this
        // synthetic 1-mode-per-summand case (because mod-3 of indices
        // 0..5 is also 0,1,2,0,1,2), but distinguishable by class
        // origin: class 1 contains {1, 4} (Wilson-fix) vs the post-
        // fallback up_quark which would be by_class[1].clone() while
        // by_class[1] is empty in the legacy bug → fallback writes
        // [0, 3]. So the assertion `up = {1, 4}` discriminates the
        // two scenarios cleanly.
        let mut up_sorted = sectors.up_quark.clone();
        up_sorted.sort_unstable();
        let mut down_sorted = sectors.down_quark.clone();
        down_sorted.sort_unstable();
        let mut lep_sorted = sectors.lepton.clone();
        lep_sorted.sort_unstable();

        assert!(
            !sectors.up_quark.is_empty(),
            "up_quark empty — legacy all-zero collapse regression"
        );
        assert!(
            !sectors.down_quark.is_empty(),
            "down_quark empty — legacy all-zero collapse regression"
        );
        assert!(
            !sectors.lepton.is_empty(),
            "lepton empty — legacy all-zero collapse regression"
        );

        // Verify the SU(3)-flavour-by-index projection: class = dom % 3.
        assert_eq!(up_sorted, vec![1, 4], "up_quark = class 1 = indices {{1,4}}");
        assert_eq!(down_sorted, vec![2, 5], "down_quark = class 2 = indices {{2,5}}");
        assert_eq!(lep_sorted, vec![0, 3], "lepton = class 0 = indices {{0,3}}");

        // At least 2 distinct phase classes — the headline assertion
        // motivating this test.
        let mut classes_seen = std::collections::HashSet::new();
        if !sectors.up_quark.is_empty() { classes_seen.insert(1); }
        if !sectors.down_quark.is_empty() { classes_seen.insert(2); }
        if !sectors.lepton.is_empty() { classes_seen.insert(0); }
        assert!(
            classes_seen.len() >= 2,
            "Wilson-fix MUST produce >= 2 distinct phase classes on \
             the AKLP 2-factor bundle; got {} classes",
            classes_seen.len()
        );
    }
}
