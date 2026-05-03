//! **Status: research-only as of commit 91a6c976.** The chain-match
//! discrimination channel using these modules is not a converged
//! discriminator — see references/p6_3_chain_match.md for the
//! retraction. Modules are kept exported for follow-up work but
//! should not be treated as a production discrimination channel.
//!
//! # Substrate-mass-chain matcher
//!
//! Given the lowest few eigenvalues of the metric Laplacian on a CY3
//! candidate (produced by [`crate::route34::metric_laplacian`]), find
//! the best assignment of eigenvalue indices to the predicted
//! Coxeter / D_8 chain positions and report the residual log-eigenvalue
//! distance.
//!
//! Per the journal entry
//! `book/journal/2026-04-29/2026-04-29-charged-fermion-spectrum-from-e8-sub-coxeter-structure.adoc`:
//!
//! - **Lepton chain**: charged-lepton substrate masses sit at φ^k for
//!   k in the E_8 Coxeter exponents `{1, 7, 11, 13, 17, 19, 23, 29}`.
//! - **Quark chain**: quark substrate masses sit at (√2)^k for k in
//!   the D_8 integer/half-integer steps `{4, 6.5, 15, 22.5, 26, 36.5}`.
//!
//! At a Donaldson-balanced metric, the substrate-standing-wave-mode
//! interpretation predicts that consecutive Laplace-Beltrami eigenvalue
//! ratios match the chain ratios. The match residual:
//!
//!     residual = Σ_k | log(λ_assigned[k]) - k · log(harmonic) |
//!
//! is computed at high precision via
//! `pwos_math::precision::BigFloat` (under the `precision-bigfloat`
//! feature) so the comparator is unaffected by f64 rounding in the
//! upstream Galerkin solve.
//!
//! The lower the residual, the better the candidate fits the predicted
//! chain. This is the **discrimination channel**.

// Suppress deprecation warnings within this module — the deprecation
// annotations on `ChainType`, `ChainMatchResult`, and `match_chain` are
// intended for downstream consumers. Internal references inside the
// module body are not the audience.
#![allow(deprecated)]
use serde::{Deserialize, Serialize};

#[cfg(feature = "precision-bigfloat")]
use pwos_math::precision::{
    bigfloat_from_str, bits_for_decimal_digits, constants, BigFloat,
};

/// Which chain to match against.
#[deprecated(
    note = "Result retracted per P6.3b — chain match is not converged in basis size; \
            see references/p6_3_chain_match.md. Type retained only for diagnostic \
            re-runs (`p6_3_chain_match_diagnostic`)."
)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChainType {
    /// φ-chain at E_8 Coxeter exponents.
    Lepton,
    /// (√2)-chain at D_8 integer / half-integer positions.
    Quark,
}

impl Default for ChainType {
    fn default() -> Self {
        ChainType::Quark
    }
}

impl ChainType {
    /// Predicted exponents on the chain. Float-typed so the quark chain
    /// can carry half-integer steps (e.g. 6.5).
    pub fn predicted_exponents(self) -> Vec<f64> {
        match self {
            ChainType::Lepton => vec![1.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0],
            ChainType::Quark => vec![4.0, 6.5, 15.0, 22.5, 26.0, 36.5],
        }
    }

    /// The harmonic ratio: φ for leptons, √2 for quarks.
    pub fn harmonic_f64(self) -> f64 {
        match self {
            ChainType::Lepton => (1.0_f64 + 5.0f64.sqrt()) / 2.0,
            ChainType::Quark => 2.0f64.sqrt(),
        }
    }
}

/// Result of [`match_chain`].
#[deprecated(
    note = "Result retracted per P6.3b — chain match is not converged in basis size; \
            see references/p6_3_chain_match.md. Type retained only for diagnostic \
            re-runs (`p6_3_chain_match_diagnostic`)."
)]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ChainMatchResult {
    /// Decimal string of the high-precision residual (so the result
    /// can serialise without dragging the `precision-bigfloat` feature
    /// into downstream consumers). When `precision-bigfloat` is off,
    /// this carries the f64 residual formatted to 17 decimal digits.
    pub residual_log_str: String,
    /// f64 truncation of the residual for fast comparisons / logging.
    pub residual_log_f64: f64,
    /// Number of predicted chain positions matched to an eigenvalue.
    pub n_matched: usize,
    /// Maximum |log(λ_k) - k_pred · log(harmonic)| across matched
    /// pairs, in f64. Useful for spotting outliers in the assignment.
    pub max_individual_dev_f64: f64,
    /// Per-chain-index assigned eigenvalue (f64). `assigned_eigvals[j]`
    /// is the eigenvalue assigned to the `j`-th predicted chain
    /// position (`predicted_exponents()[j]`). `NaN` when no eigenvalue
    /// could be assigned (e.g. running out of eigenvalues).
    pub assigned_eigvals: Vec<f64>,
    /// Predicted chain exponents (copy of `chain_type.predicted_exponents()`).
    pub predicted_exponents: Vec<f64>,
    /// Chain type used.
    pub chain_type: ChainType,
}

/// Greedy nearest-neighbour assignment in log-eigenvalue space.
///
/// **Note (P7.2)**: this greedy implementation is preserved for
/// regression / parity testing. Production callers should use
/// [`match_chain_hungarian`], which performs an *optimal* assignment
/// in log-eigenvalue space (Hungarian / Jonker-Volgenant). Round-2
/// hostile review of P6.3 noted that greedy NN is suboptimal vs the
/// Hungarian solution: greedy assigns one chain position at a time
/// without revisiting earlier choices, so an early greedy pick can
/// "lock out" a globally cheaper assignment of a later position.
/// Hungarian minimises the *total* squared log-residual.
///
/// Algorithm (greedy):
/// 1. Filter out non-positive / non-finite eigenvalues (drop the
///    `λ_0 ≈ 0` constant mode and any spurious zeros).
/// 2. Compute `log(λ_0)` from the lowest remaining eigenvalue and treat
///    it as the chain origin. Remaining log-eigenvalues are
///    `δ_n := log(λ_n) - log(λ_0)`.
/// 3. For each predicted chain position `k_pred` (sorted ascending),
///    the predicted log-distance is `(k_pred - k_0) · log(harmonic)`
///    where `k_0 = predicted_exponents[0]`. Find the unassigned
///    eigenvalue index whose `δ_n` is closest to the predicted
///    log-distance.
/// 4. Sum `|δ_assigned - δ_predicted|` across all matched positions —
///    the chain-match residual. High precision via BigFloat when
///    `precision-bigfloat` is enabled.
#[deprecated(
    note = "Result retracted per P6.3b — chain match is not converged in basis size; \
            see references/p6_3_chain_match.md. Function retained only for diagnostic \
            re-runs (`p6_3_chain_match_diagnostic`)."
)]
pub fn match_chain(eigenvalues: &[f64], chain_type: ChainType) -> ChainMatchResult {
    let predicted = chain_type.predicted_exponents();
    let mut result = ChainMatchResult {
        chain_type,
        predicted_exponents: predicted.clone(),
        assigned_eigvals: vec![f64::NAN; predicted.len()],
        ..Default::default()
    };

    // Filter to positive eigenvalues (drop constant mode + numerical
    // zeros). Maintain ascending order.
    let mut positive: Vec<f64> = eigenvalues
        .iter()
        .cloned()
        .filter(|&v| v.is_finite() && v > 1.0e-9)
        .collect();
    positive.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if positive.is_empty() {
        result.residual_log_str = "Inf".to_string();
        result.residual_log_f64 = f64::INFINITY;
        return result;
    }

    let log_lambda_0 = positive[0].ln();
    let log_h = chain_type.harmonic_f64().ln();

    // Predicted log-distance from the chain origin (k_0 = predicted[0]).
    let k_0 = predicted[0];
    let predicted_logd: Vec<f64> = predicted
        .iter()
        .map(|&k| (k - k_0) * log_h)
        .collect();

    // Available eigenvalue slots (skip lambda_0 itself for the
    // origin: assigned_eigvals[0] = positive[0]).
    let mut taken = vec![false; positive.len()];
    taken[0] = true;
    result.assigned_eigvals[0] = positive[0];

    let mut max_dev = 0.0f64;
    let mut sum_residual_f64 = 0.0f64;

    for j in 1..predicted_logd.len() {
        let target = predicted_logd[j];
        // Find unassigned positive[k] with |log(positive[k]) - log_lambda_0 - target| minimal.
        let mut best_idx: Option<usize> = None;
        let mut best_dev = f64::INFINITY;
        for k in 0..positive.len() {
            if taken[k] {
                continue;
            }
            let delta = positive[k].ln() - log_lambda_0;
            let dev = (delta - target).abs();
            if dev < best_dev {
                best_dev = dev;
                best_idx = Some(k);
            }
        }
        if let Some(k) = best_idx {
            taken[k] = true;
            result.assigned_eigvals[j] = positive[k];
            sum_residual_f64 += best_dev;
            if best_dev > max_dev {
                max_dev = best_dev;
            }
            result.n_matched += 1;
        }
    }
    // Origin counts as matched too.
    result.n_matched += 1;
    result.max_individual_dev_f64 = max_dev;

    // High-precision residual.
    #[cfg(feature = "precision-bigfloat")]
    {
        let prec = bits_for_decimal_digits(150).max(502);
        let log_h_bf = match chain_type {
            ChainType::Lepton => log_bigfloat(constants::phi(prec), prec),
            ChainType::Quark => log_bigfloat(constants::sqrt2(prec), prec),
        };
        let mut acc = bigfloat_from_str("0", prec);
        for j in 0..predicted_logd.len() {
            let lam = result.assigned_eigvals[j];
            if !lam.is_finite() || lam <= 0.0 {
                continue;
            }
            let delta_bf = log_bigfloat(
                bigfloat_from_str(&format_f64(lam), prec)
                    .div(&bigfloat_from_str(&format_f64(positive[0]), prec), prec, ROUND),
                prec,
            );
            let k_diff = predicted[j] - k_0;
            let target_bf = log_h_bf
                .mul(&bigfloat_from_str(&format_f64(k_diff), prec), prec, ROUND);
            let diff = delta_bf.sub(&target_bf, prec, ROUND);
            let abs_diff = if is_neg(&diff) {
                diff.neg()
            } else {
                diff
            };
            acc = acc.add(&abs_diff, prec, ROUND);
        }
        result.residual_log_str = pwos_math::precision::bigfloat_to_string(&acc);
        result.residual_log_f64 = sum_residual_f64;
    }
    #[cfg(not(feature = "precision-bigfloat"))]
    {
        result.residual_log_str = format!("{:.17e}", sum_residual_f64);
        result.residual_log_f64 = sum_residual_f64;
    }
    result
}

#[cfg(feature = "precision-bigfloat")]
const ROUND: astro_float::RoundingMode = astro_float::RoundingMode::ToEven;

#[cfg(feature = "precision-bigfloat")]
fn format_f64(v: f64) -> String {
    // 17-digit decimal round-trip — sufficient for f64.
    format!("{:.17e}", v)
}

#[cfg(feature = "precision-bigfloat")]
fn is_neg(b: &BigFloat) -> bool {
    b.is_negative()
}

#[cfg(feature = "precision-bigfloat")]
fn log_bigfloat(x: BigFloat, prec: usize) -> BigFloat {
    // astro-float exposes ln on BigFloat.
    let mut cc = astro_float::Consts::new().expect("astro-float Consts");
    x.ln(prec, ROUND, &mut cc)
}

// ---------------------------------------------------------------------
// P7.2 — Hungarian (optimal) assignment in log-eigenvalue space.
//
// The cost matrix C[i][j] = (log(λ_i) - k_j · log(harmonic))² (squared
// log-residual). With at most n ≈ 50 eigenvalues vs ≤ 8 chain
// positions, a square-padded O(n³) Hungarian solve is sub-millisecond.
// We implement the O(n³) Kuhn-Munkres / Jonker-Volgenant variant using
// the standard potential-function update; padding to a square matrix
// with sentinel-large costs handles the rectangular case (more
// eigenvalues than chain slots).
// ---------------------------------------------------------------------

/// Solve a rectangular assignment problem (rows ≤ cols) minimising
/// `Σ C[r][col[r]]`. Returns `assignment[r] = c` for r in 0..n_rows,
/// where `c < n_cols` is the column assigned to row r.
///
/// O(n³) Hungarian with row potentials. For our use case n_rows is
/// the number of chain slots (≤ 8) and n_cols is the number of
/// eigenvalues (~30-50), so cost is negligible. Costs must be finite,
/// non-negative, and row-major (`cost[r * n_cols + c]`).
pub fn hungarian_assign(cost: &[f64], n_rows: usize, n_cols: usize) -> Vec<usize> {
    assert!(n_rows <= n_cols, "hungarian_assign: n_rows ({}) must be <= n_cols ({})", n_rows, n_cols);
    if n_rows == 0 {
        return Vec::new();
    }
    // Square-pad: extend rows with zero-cost dummies, so the standard
    // square Hungarian applies. We only read assignments for the
    // original `n_rows`.
    let n = n_cols;
    let mut c = vec![0.0f64; n * n];
    for r in 0..n_rows {
        for j in 0..n_cols {
            c[r * n + j] = cost[r * n_cols + j];
        }
    }
    // Dummy rows are already zero-filled.

    // Jonker-Volgenant style with O(n³) augmenting-path search.
    // Indices below: u[i] potentials on rows, v[j] on cols, p[j] is
    // the row matched to column j (0 means unmatched in the working
    // index scheme; we shift by 1).
    let inf = f64::INFINITY;
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; n + 1];
    let mut p = vec![0usize; n + 1];
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=n {
                if !used[j] {
                    let cur = c[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=n {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // p[j] gives the row matched to column j (1-indexed). Invert.
    let mut ans = vec![usize::MAX; n_rows];
    for j in 1..=n {
        let r = p[j];
        if r >= 1 && (r - 1) < n_rows {
            ans[r - 1] = j - 1;
        }
    }
    // Every original row should have been assigned (square pad
    // guarantees this).
    for r in 0..n_rows {
        debug_assert!(ans[r] != usize::MAX, "hungarian_assign: row {} unassigned", r);
    }
    ans
}

/// Hungarian / optimal-assignment variant of [`match_chain`].
///
/// Builds the squared-log cost matrix `C[j][i] = (log(λ_i) - k_j · log(h))²`
/// over positive eigenvalues vs predicted chain exponents, solves the
/// rectangular assignment minimising `Σ_j C[j][assigned(j)]`, then
/// reports the absolute-log residual `Σ_j |log(λ_i) - k_j · log(h)|`
/// at the optimal assignment.
///
/// Note: we minimise squared residual (smooth, gradient-friendly) but
/// report the |·| residual to keep the same scale and units as the
/// greedy implementation. The optimal assignment under squared cost is
/// the optimal assignment under |·| cost as long as no two costs tie
/// (and ties don't change the optimum value, only the realising
/// assignment).
///
/// Unlike the greedy version, this does NOT pin the smallest
/// eigenvalue to the chain origin. The Hungarian solver discovers the
/// best origin alignment as part of the optimal assignment.
#[deprecated(
    note = "Result retracted per P6.3b — chain match is not converged in basis size; \
            see references/p6_3_chain_match.md. Function retained only for diagnostic \
            re-runs (`p6_3_chain_match_diagnostic`)."
)]
pub fn match_chain_hungarian(
    eigenvalues: &[f64],
    chain_type: ChainType,
) -> ChainMatchResult {
    let predicted = chain_type.predicted_exponents();
    let mut result = ChainMatchResult {
        chain_type,
        predicted_exponents: predicted.clone(),
        assigned_eigvals: vec![f64::NAN; predicted.len()],
        ..Default::default()
    };

    let mut positive: Vec<f64> = eigenvalues
        .iter()
        .cloned()
        .filter(|&v| v.is_finite() && v > 1.0e-9)
        .collect();
    positive.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if positive.len() < predicted.len() {
        // Insufficient eigenvalues to match the chain — return Inf.
        result.residual_log_str = "Inf".to_string();
        result.residual_log_f64 = f64::INFINITY;
        return result;
    }

    let log_h = chain_type.harmonic_f64().ln();
    // Anchor: align the chain origin to the smallest log eigenvalue,
    // matching the convention of [`match_chain`]. Without an anchor
    // there is a global log-shift gauge freedom; pinning λ_min to
    // exp(k_0 · log h) fixes the gauge and makes the residual
    // well-defined and comparable across candidates.
    let log_lambda_0 = positive[0].ln();
    let k_0 = predicted[0];

    // Cost matrix: rows = chain slots (n_rows), cols = eigenvalues
    // (n_cols). C[j][i] = (δ_i - target_j)² where δ_i = log(λ_i) -
    // log(λ_0) and target_j = (k_j - k_0) · log(h).
    let n_rows = predicted.len();
    let n_cols = positive.len();
    let mut cost = vec![0.0f64; n_rows * n_cols];
    let logs: Vec<f64> = positive.iter().map(|v| v.ln() - log_lambda_0).collect();
    for j in 0..n_rows {
        let target = (predicted[j] - k_0) * log_h;
        for i in 0..n_cols {
            let d = logs[i] - target;
            cost[j * n_cols + i] = d * d;
        }
    }
    let assign = hungarian_assign(&cost, n_rows, n_cols);

    let mut sum_abs = 0.0f64;
    let mut max_dev = 0.0f64;
    for j in 0..n_rows {
        let i = assign[j];
        let lam = positive[i];
        result.assigned_eigvals[j] = lam;
        let target = (predicted[j] - k_0) * log_h;
        let dev = (logs[i] - target).abs();
        sum_abs += dev;
        if dev > max_dev {
            max_dev = dev;
        }
    }
    result.n_matched = n_rows;
    result.max_individual_dev_f64 = max_dev;

    // High-precision residual (mirror the BigFloat path of greedy).
    #[cfg(feature = "precision-bigfloat")]
    {
        let prec = bits_for_decimal_digits(150).max(502);
        let log_h_bf = match chain_type {
            ChainType::Lepton => log_bigfloat(constants::phi(prec), prec),
            ChainType::Quark => log_bigfloat(constants::sqrt2(prec), prec),
        };
        let mut acc = bigfloat_from_str("0", prec);
        for j in 0..n_rows {
            let lam = result.assigned_eigvals[j];
            if !lam.is_finite() || lam <= 0.0 {
                continue;
            }
            let delta_bf = log_bigfloat(
                bigfloat_from_str(&format_f64(lam), prec)
                    .div(&bigfloat_from_str(&format_f64(positive[0]), prec), prec, ROUND),
                prec,
            );
            let k_diff = predicted[j] - k_0;
            let target_bf = log_h_bf
                .mul(&bigfloat_from_str(&format_f64(k_diff), prec), prec, ROUND);
            let diff = delta_bf.sub(&target_bf, prec, ROUND);
            let abs_diff = if is_neg(&diff) { diff.neg() } else { diff };
            acc = acc.add(&abs_diff, prec, ROUND);
        }
        result.residual_log_str = pwos_math::precision::bigfloat_to_string(&acc);
        result.residual_log_f64 = sum_abs;
    }
    #[cfg(not(feature = "precision-bigfloat"))]
    {
        result.residual_log_str = format!("{:.17e}", sum_abs);
        result.residual_log_f64 = sum_abs;
    }
    result
}

// ---------------------------------------------------------------------
// P7.2 — Eigenvalue-ratio pattern analysis.
//
// Round-1 hostile review noted that chain ratios span O(10^5) while
// the f64 spectrum spans much less, so absolute chain residuals are
// dominated by which eigenvalues sit at the chain extrema. The right
// comparator is the *gap structure*: do consecutive log-eigenvalue
// gaps cluster at φ^Δk or (√2)^Δk for predicted Δk in {1, 2, … 7}?
// ---------------------------------------------------------------------

/// One entry of the ratio-pattern report: a consecutive eigenvalue
/// ratio λ_{n+1}/λ_n compared against the closest predicted harmonic
/// power ratio.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RatioMatch {
    /// 0-based index n of the lower eigenvalue in the ratio.
    pub idx: usize,
    /// λ_n value.
    pub lambda_lo: f64,
    /// λ_{n+1} value.
    pub lambda_hi: f64,
    /// λ_{n+1} / λ_n (raw f64).
    pub ratio: f64,
    /// Closest predicted ratio = harmonic^Δk for the listed harmonic
    /// (string-tagged so the consumer doesn't need the chain enum).
    pub closest_harmonic: String,
    /// The Δk that produced `closest_predicted`. Half-integer for
    /// √2-chain.
    pub closest_dk: f64,
    /// `harmonic^Δk` value.
    pub closest_predicted: f64,
    /// Relative residual `(ratio - closest_predicted) / closest_predicted`
    /// — sign-bearing so the consumer can read off whether the
    /// observed gap is wider or narrower than predicted.
    pub rel_residual: f64,
}

/// Inspect the largest consecutive eigenvalue ratios in the lowest
/// `n_floor` positive eigenvalues, and check each against φ^Δk or
/// (√2)^Δk for Δk in {1, 2, 3, 4, 5, 6, 7} (and the half-integer
/// equivalents for √2).
///
/// Returns up to `n_top` `RatioMatch` entries, sorted by descending
/// raw ratio value (largest gap first).
pub fn ratio_pattern(eigenvalues: &[f64], n_floor: usize, n_top: usize) -> Vec<RatioMatch> {
    let mut positive: Vec<f64> = eigenvalues
        .iter()
        .cloned()
        .filter(|&v| v.is_finite() && v > 1.0e-9)
        .collect();
    positive.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = positive.len().min(n_floor);
    if n < 2 {
        return Vec::new();
    }
    let phi = (1.0f64 + 5.0f64.sqrt()) / 2.0;
    let s2 = 2.0f64.sqrt();
    // Build the candidate predicted-ratio table once.
    // Phi: integer Δk only (E_8 Coxeter exponents are integer).
    // Sqrt2: integer + half-integer (D_8 admits both).
    let mut targets: Vec<(String, f64, f64)> = Vec::new();
    for dk in 1..=7 {
        targets.push(("phi".to_string(), dk as f64, phi.powi(dk)));
    }
    for &half in &[
        0.5f64, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
    ] {
        targets.push(("sqrt2".to_string(), half, s2.powf(half)));
    }

    let mut all: Vec<RatioMatch> = Vec::with_capacity(n.saturating_sub(1));
    for idx in 0..(n - 1) {
        let lo = positive[idx];
        let hi = positive[idx + 1];
        if lo <= 0.0 {
            continue;
        }
        let r = hi / lo;
        // Find closest target.
        let (mut best_h, mut best_dk, mut best_pred, mut best_rel) =
            ("none".to_string(), f64::NAN, f64::NAN, f64::INFINITY);
        for (h, dk, pred) in &targets {
            let rel = (r - pred) / pred;
            if rel.abs() < best_rel.abs() {
                best_h = h.clone();
                best_dk = *dk;
                best_pred = *pred;
                best_rel = rel;
            }
        }
        all.push(RatioMatch {
            idx,
            lambda_lo: lo,
            lambda_hi: hi,
            ratio: r,
            closest_harmonic: best_h,
            closest_dk: best_dk,
            closest_predicted: best_pred,
            rel_residual: best_rel,
        });
    }
    // Sort by descending raw ratio (largest gap first).
    all.sort_by(|a, b| b.ratio.partial_cmp(&a.ratio).unwrap_or(std::cmp::Ordering::Equal));
    all.truncate(n_top);
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quark_chain_perfect_match_residual_zero() {
        // Construct a perfect (√2)-chain at the predicted exponents,
        // starting at λ_0 = 1.
        let s2 = 2.0f64.sqrt();
        let exps = ChainType::Quark.predicted_exponents();
        let k0 = exps[0];
        let eigs: Vec<f64> = exps.iter().map(|&k| s2.powf(k - k0)).collect();
        let r = match_chain(&eigs, ChainType::Quark);
        assert_eq!(r.n_matched, exps.len());
        assert!(
            r.residual_log_f64 < 1.0e-9,
            "perfect chain should give zero residual, got {}",
            r.residual_log_f64
        );
    }

    #[test]
    fn lepton_chain_off_by_factor_two_has_positive_residual() {
        // Multiply the second eigenvalue by 2 — should produce a clearly
        // non-zero residual.
        let phi = (1.0_f64 + 5.0f64.sqrt()) / 2.0;
        let exps = ChainType::Lepton.predicted_exponents();
        let k0 = exps[0];
        let mut eigs: Vec<f64> = exps.iter().map(|&k| phi.powf(k - k0)).collect();
        eigs[1] *= 2.0;
        let r = match_chain(&eigs, ChainType::Lepton);
        assert!(
            r.residual_log_f64 > 0.5,
            "perturbed chain should give residual > 0.5, got {}",
            r.residual_log_f64
        );
    }

    #[test]
    fn empty_eigvals_returns_infinite_residual() {
        let r = match_chain(&[], ChainType::Quark);
        assert!(r.residual_log_f64.is_infinite());
        assert_eq!(r.n_matched, 0);
    }

    // -----------------------------------------------------------------
    // P7.2 — Hungarian assignment regression tests.
    // -----------------------------------------------------------------

    /// Adversarial cost matrix where greedy-by-row picks a globally
    /// suboptimal assignment. Direct test on `hungarian_assign` rather
    /// than `match_chain_hungarian` (which is anchored at λ_min and
    /// thus largely 1D-monotone, where greedy is provably near-optimal).
    ///
    /// 3x3 cost matrix:
    ///   [ 0   100  100 ]
    ///   [ 100   0  100 ]
    ///   [  90  90    0 ]
    /// Greedy-by-row would pick (0,0)=0, (1,1)=0, (2,2)=0 — total 0.
    /// In this case greedy IS optimal.
    ///
    /// The classic greedy-fails matrix:
    ///   [ 1 0 ]
    ///   [ 0 0 ]
    /// Greedy-by-row first row: argmin is col 1 (cost 0). Second row:
    /// only col 0 left (cost 0). Total = 0. Hungarian: same. No
    /// difference — both pick (0,1), (1,0).
    ///
    /// Real failing case:
    ///   [ 5 10 ]
    ///   [ 1 6 ]
    /// Greedy row 0: col 0 (cost 5). Row 1: col 1 (cost 6). Total 11.
    /// Hungarian: row 0 → col 1 (cost 10), row 1 → col 0 (cost 1).
    /// Total 11. Same.
    ///
    /// The bigger gap is when greedy has different sort priorities:
    ///   [  3  100   2 ]
    ///   [  4   1   100 ]
    ///   [100   2    1 ]
    /// "Greedy by row, take min unassigned col":
    ///   row 0: min over [0,1,2] is col 2 (cost 2).
    ///   row 1: min over [0,1] is col 1 (cost 1).
    ///   row 2: only col 0 left (cost 100).
    ///   Total = 103.
    /// Hungarian: row 0 → col 0 (3), row 1 → col 1 (1), row 2 → col 2 (1).
    ///   Total = 5. <<< much better.
    ///
    /// We use this 3x3 to verify Hungarian's optimum directly.
    #[test]
    fn hungarian_beats_greedy_on_adversarial_costs() {
        let cost = vec![
            3.0, 100.0, 2.0,
            4.0, 1.0, 100.0,
            100.0, 2.0, 1.0,
        ];
        // Greedy-by-row reference.
        let mut taken = vec![false; 3];
        let mut greedy_total = 0.0f64;
        for r in 0..3 {
            let mut best_c = usize::MAX;
            let mut best_v = f64::INFINITY;
            for c in 0..3 {
                if taken[c] {
                    continue;
                }
                if cost[r * 3 + c] < best_v {
                    best_v = cost[r * 3 + c];
                    best_c = c;
                }
            }
            taken[best_c] = true;
            greedy_total += best_v;
        }
        let hung = hungarian_assign(&cost, 3, 3);
        let hung_total: f64 = (0..3).map(|r| cost[r * 3 + hung[r]]).sum();

        assert!(
            hung_total <= greedy_total + 1e-9,
            "Hungarian total ({}) must be ≤ greedy ({})",
            hung_total,
            greedy_total
        );
        // For this specific construction, Hungarian beats greedy by ~98.
        assert!(
            hung_total < greedy_total - 50.0,
            "Hungarian should strictly beat greedy by > 50 (greedy={}, hung={})",
            greedy_total,
            hung_total
        );
    }

    /// Parity test on the chain-match path: under the chain anchor
    /// (λ_min pinned to slot 0), Hungarian's residual is always ≤
    /// greedy's. We don't require strictly better — under monotone 1D
    /// cost they agree — only that Hungarian never regresses.
    #[test]
    fn hungarian_never_regresses_vs_greedy_on_chain() {
        let s2 = 2.0f64.sqrt();
        let exps = ChainType::Quark.predicted_exponents();
        let k0 = exps[0];
        // Perfect chain plus a few decoys.
        let mut eigs: Vec<f64> = exps.iter().map(|&k| s2.powf(k - k0)).collect();
        eigs.push(s2.powf(40.0 - k0));
        eigs.push(s2.powf(20.0 - k0));
        eigs.push(s2.powf(8.0 - k0));
        let r_g = match_chain(&eigs, ChainType::Quark);
        let r_h = match_chain_hungarian(&eigs, ChainType::Quark);
        assert!(
            r_h.residual_log_f64 <= r_g.residual_log_f64 + 1e-9,
            "Hungarian ({}) should be ≤ greedy ({}) under chain anchor",
            r_h.residual_log_f64,
            r_g.residual_log_f64
        );
    }

    /// Hungarian matches greedy exactly on a perfect chain (no decoys,
    /// no adversarial layout) — establishes parity in the
    /// non-pathological case.
    #[test]
    fn hungarian_matches_greedy_on_perfect_chain() {
        let s2 = 2.0f64.sqrt();
        let exps = ChainType::Quark.predicted_exponents();
        let k0 = exps[0];
        let eigs: Vec<f64> = exps.iter().map(|&k| s2.powf(k - k0)).collect();
        let r_g = match_chain(&eigs, ChainType::Quark);
        let r_h = match_chain_hungarian(&eigs, ChainType::Quark);
        assert!(r_g.residual_log_f64 < 1e-9);
        assert!(r_h.residual_log_f64 < 1e-9);
        assert_eq!(r_h.n_matched, exps.len());
    }

    /// Tiny smoke test on the underlying `hungarian_assign`: the
    /// classic 3x3 example from any algorithms text.
    #[test]
    fn hungarian_assign_3x3() {
        // Minimum-cost assignment for
        //   [ 4 1 3 ]
        //   [ 2 0 5 ]
        //   [ 3 2 2 ]
        // is rows -> cols (0->1, 1->0, 2->2) with cost 1+2+2 = 5.
        let cost = vec![4.0, 1.0, 3.0, 2.0, 0.0, 5.0, 3.0, 2.0, 2.0];
        let a = hungarian_assign(&cost, 3, 3);
        let total: f64 = (0..3).map(|r| cost[r * 3 + a[r]]).sum();
        assert!(
            (total - 5.0).abs() < 1e-9,
            "Hungarian 3x3: expected cost 5 (0->1, 1->0, 2->2 or similar), got {} (assign={:?})",
            total,
            a
        );
    }

    /// Ratio-pattern: a clean (√2)-chain produces predictable
    /// consecutive-ratio matches.
    #[test]
    fn ratio_pattern_clean_sqrt2_chain() {
        let s2 = 2.0f64.sqrt();
        // Build λ at k=0,1,2,3,4,5 — every consecutive ratio = √2,
        // matching closest_dk = 1.0 with rel_residual ≈ 0.
        let eigs: Vec<f64> = (0..6).map(|k| s2.powi(k)).collect();
        let pat = ratio_pattern(&eigs, 30, 5);
        assert!(!pat.is_empty());
        for m in &pat {
            assert!(m.rel_residual.abs() < 1e-9, "expected exact match, got {:?}", m);
            assert_eq!(m.closest_harmonic, "sqrt2");
            assert!((m.closest_dk - 1.0).abs() < 1e-9);
        }
    }
}
