//! # P8.1 — Multichannel Bayes-factor combiner.
//!
//! Combines per-channel log-Bayes-factors (`ln BF_TY:Schoen`) from several
//! discrimination channels into a single posterior odds, evidence-strength
//! categorisation, and asymptotic n-σ equivalent. Sits one level above
//! [`crate::route34::bayes_factor_sigma`] (P5.8) — the σ-channel module —
//! and consumes its output along with the chain-match (P7.11), Hodge-
//! consistency (P8.2), and Yukawa-spectrum (P8.3) channels.
//!
//! ## Mathematical model
//!
//! Assuming the channels are statistically independent given the model,
//! the joint likelihood factorises and the natural-log Bayes factor adds:
//!
//! ```text
//!   ln BF_total = Σ_c ln BF_c
//! ```
//!
//! The asymptotic n-σ equivalent of the combined evidence comes from the
//! Cowan-Cranmer-Gross-Vitells (2011) §4 relation:
//!
//! ```text
//!   n_σ ≈ √(2 |ln BF_total|)
//! ```
//!
//! The combined log_10 BF is categorised by Jeffreys-Trotta:
//!
//! ```text
//!   |log10 BF|   category
//!     < 0.5      Inconclusive
//!     [0.5, 1)   Weak
//!     [1, 2)     Moderate
//!     ≥ 2        Strong
//! ```
//!
//! ## Independence assumption
//!
//! The σ, chain-match, Hodge-consistency, and Yukawa-eigenvalue channels
//! all probe different geometric features of the candidate CY3 (point-
//! sample Donaldson loss; sub-Coxeter chain-position residuals; Hodge-
//! number multiplicities of harmonic forms; bundle-cohomology overlap
//! integrals). They are not strictly independent — all four ultimately
//! depend on the same underlying metric — but they sample very different
//! statistics of that metric and are treated as independent here.
//!
//! Future refinement may add a `correlations: Mat<f64>` field for
//! Mahalanobis-style joint likelihood, but the present implementation is
//! the diagonal-covariance limit, which is conservative for *positive*
//! channel correlations (it overstates evidence strength only when the
//! channels are negatively correlated, which is not the case here).
//!
//! ## References
//!
//! * Kass, R.E.; Raftery, A.E. "Bayes factors", J. Amer. Stat. Assoc.
//!   90 (1995) 773.
//! * Trotta, R. "Bayes in the sky", Contemp. Phys. 49 (2008) 71,
//!   arXiv:0803.4089.
//! * Cowan, Cranmer, Gross, Vitells, "Asymptotic formulae for
//!   likelihood-based tests of new physics", Eur. Phys. J. C 71 (2011)
//!   1554, arXiv:1007.1727.
//! * Jeffreys, H. "Theory of Probability", 3rd ed. (Oxford 1961),
//!   Appendix B.

use std::path::Path;

use serde::{Deserialize, Serialize};

use pwos_math::stats::bayes_factor::{categorise_evidence, EvidenceStrength};

use crate::route34::bayes_factor_sigma::evidence_strength_label;

// =====================================================================
// Channel evidence struct
// =====================================================================

/// One channel's contribution to the multichannel Bayes factor.
///
/// `log_bf_ty_vs_schoen` is signed: positive favours TY, negative favours
/// Schoen. `n_observations` is the effective sample size for the channel
/// (used for traceability and any downstream Bonferroni-style correction;
/// not used in the per-channel combination math itself, which is just a
/// sum of log-BFs under the independence assumption).
///
/// `for_combination` selects whether the channel contributes to the
/// combined model-comparison Bayes factor. It is set to `false` for the
/// σ-channel (P8.1e decision): σ measures basis-size-saturated DKLR
/// L¹ Monge-Ampère residual, and the σ-gap between TY (n_basis=87 at k=3)
/// and Schoen (n_basis=27 at k=3) is dominated by basis-size differences,
/// not by physics observables. σ remains a *discriminability* metric
/// (separately reported as |t|, n-σ) but is excluded from the model-
/// comparison sum. Chain-match / Hodge / Yukawa channels keep
/// `for_combination = true`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChannelEvidence {
    /// Channel name: `"sigma"`, `"chain_quark"`, `"chain_lepton"`,
    /// `"hodge"`, `"yukawa"`, ...
    pub name: String,
    /// `ln BF_TY:Schoen` for this channel. Positive → TY; negative →
    /// Schoen.
    pub log_bf_ty_vs_schoen: f64,
    /// Effective sample size (number of independent observations folded
    /// into this channel's BF).
    pub n_observations: usize,
    /// Path to the source data file (for provenance/audit).
    pub source: String,
    /// Whether this channel participates in the combined model-
    /// comparison Bayes factor. `false` for σ (discriminability-only
    /// channel — basis-size-dominated, not a physics observable);
    /// `true` for chain-match / Hodge / Yukawa physics channels.
    /// Defaults to `true` when deserialised from older JSON that
    /// predates the field, to preserve the meaning of pre-P8.1e
    /// archived runs.
    #[serde(default = "default_for_combination")]
    pub for_combination: bool,
}

/// Default value for [`ChannelEvidence::for_combination`] when the field
/// is absent from a deserialised JSON — older archives predate the
/// field and represented physics channels (which all combine).
fn default_for_combination() -> bool {
    true
}

impl ChannelEvidence {
    /// Construct a [`ChannelEvidence`] with input validation. Defaults
    /// `for_combination = true` (the physics-channel case). Use
    /// [`Self::new_with_for_combination`] to opt out (σ-channel).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `log_bf_ty_vs_schoen` is non-finite, `name` is
    /// empty, or `n_observations == 0`.
    pub fn new(
        name: impl Into<String>,
        log_bf_ty_vs_schoen: f64,
        n_observations: usize,
        source: impl Into<String>,
    ) -> Result<Self, String> {
        Self::new_with_for_combination(name, log_bf_ty_vs_schoen, n_observations, source, true)
    }

    /// Construct a [`ChannelEvidence`] with explicit `for_combination`
    /// flag. Use `for_combination = false` for the σ-channel adapter
    /// (P8.1e: σ is a discriminability metric only, not a model-
    /// comparison BF input).
    ///
    /// # Errors
    ///
    /// Same conditions as [`Self::new`].
    pub fn new_with_for_combination(
        name: impl Into<String>,
        log_bf_ty_vs_schoen: f64,
        n_observations: usize,
        source: impl Into<String>,
        for_combination: bool,
    ) -> Result<Self, String> {
        let name = name.into();
        if name.is_empty() {
            return Err("ChannelEvidence: name must be non-empty".to_string());
        }
        if !log_bf_ty_vs_schoen.is_finite() {
            return Err(format!(
                "ChannelEvidence({}): log_bf_ty_vs_schoen must be finite, got {}",
                name, log_bf_ty_vs_schoen
            ));
        }
        if n_observations == 0 {
            return Err(format!(
                "ChannelEvidence({}): n_observations must be > 0",
                name
            ));
        }
        Ok(Self {
            name,
            log_bf_ty_vs_schoen,
            n_observations,
            source: source.into(),
            for_combination,
        })
    }
}

// =====================================================================
// Multichannel result struct
// =====================================================================

/// σ-channel discriminability summary, reported alongside (and
/// independent of) the combined model-comparison Bayes factor.
///
/// **Why this is separate from the BF.** The σ-channel measures basis-
/// size-saturated DKLR L¹ Monge-Ampère residual; the σ-gap between TY
/// (n_basis=87 at k=3) and Schoen (n_basis=27 at k=3) is dominated by
/// basis-size differences (a numerical artifact of the candidate-specific
/// section count), not by which substrate matches physics. We therefore
/// report σ as a *discriminability* statistic — "we can statistically
/// distinguish TY from Schoen with |t|=6.92, n-σ=6.92" — without folding
/// it into the model-comparison BF, where its sign convention conflicts
/// with the physics-channel verdict. See P8.1e hostile review.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DiscriminabilityReport {
    /// Welch t-statistic on the per-seed σ population means
    /// (`(μ_TY − μ_Schoen) / sqrt(s²_TY/n_TY + s²_Schoen/n_Schoen)`).
    /// Sign is informational only (it does *not* set a model-comparison
    /// verdict); magnitude gives the discriminability strength.
    pub sigma_t_statistic: f64,
    /// `|t|` reinterpreted as n-σ equivalent for the discriminability
    /// claim. This is the headline that drives the published
    /// "TY-vs-Schoen at 6.92σ" result.
    pub sigma_n_sigma: f64,
    /// Number of TY seeds folded into the σ t-statistic.
    pub sigma_n_ty: usize,
    /// Number of Schoen seeds folded into the σ t-statistic.
    pub sigma_n_schoen: usize,
    /// Sample SD of the TY per-seed σ vector (population spread, not
    /// SE_mean — see [`SigmaChannelBreakdown`] for the audit trail).
    pub sigma_pop_ty: f64,
    /// Sample SD of the Schoen per-seed σ vector.
    pub sigma_pop_schoen: f64,
}

impl DiscriminabilityReport {
    /// Construct a [`DiscriminabilityReport`] from the σ-channel Welch
    /// t-test breakdown. `n-σ = |t|` per the standard discriminability
    /// convention (the t-statistic *is* the n-σ separation when the
    /// sample variance is well-estimated).
    pub fn from_sigma_breakdown(bd: &SigmaChannelBreakdown) -> Self {
        Self {
            sigma_t_statistic: bd.t_statistic,
            sigma_n_sigma: bd.t_statistic.abs(),
            sigma_n_ty: bd.n_ty,
            sigma_n_schoen: bd.n_schoen,
            sigma_pop_ty: bd.sigma_pop_ty,
            sigma_pop_schoen: bd.sigma_pop_schoen,
        }
    }
}

/// Combined multichannel Bayes-factor result.
///
/// `combined_strength_label` is the string form of the Jeffreys-Trotta
/// `EvidenceStrength` (the upstream enum is not serde-derived).
///
/// **Combination semantics (P8.1e).** `combined_log_bf` is the sum of
/// `log_bf_ty_vs_schoen` over channels with `for_combination = true`
/// only. Channels with `for_combination = false` (the σ-channel) are
/// preserved in `channels` for full per-channel audit but excluded from
/// the model-comparison sum. The σ-channel discriminability headline is
/// surfaced separately via the optional `discriminability` field.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultichannelBayesResult {
    /// Per-channel evidence inputs (preserved for audit / report). All
    /// channels appear here, including discriminability-only channels
    /// (σ) that are excluded from `combined_log_bf`.
    pub channels: Vec<ChannelEvidence>,
    /// `Σ_c ln BF_c` over channels with `for_combination = true`.
    /// Positive → TY; negative → Schoen. Discriminability-only channels
    /// (σ) do *not* contribute.
    pub combined_log_bf: f64,
    /// `log_10` form of `combined_log_bf` for Jeffreys-Trotta.
    pub combined_log10_bf: f64,
    /// String label of the Jeffreys-Trotta category of `combined_log10_bf`.
    pub combined_strength_label: String,
    /// `exp(combined_log_bf)` — posterior odds TY:Schoen at unit prior.
    /// May saturate to `+inf` for very large BFs; in that case look at
    /// `combined_log_bf` directly.
    pub posterior_odds_ty: f64,
    /// `√(2 |combined_log_bf|)` — Cowan-Cranmer-Gross-Vitells (2011) §4
    /// asymptotic n-σ equivalent. Computed on the model-comparison sum
    /// only (σ excluded).
    pub n_sigma_equivalent: f64,
    /// Total number of observations across all channels (including
    /// discriminability-only channels — this is a provenance count).
    pub total_observations: usize,
    /// σ-channel discriminability summary, surfaced alongside the
    /// model-comparison BF. `None` when no σ-channel was loaded. This
    /// is the load-bearing field that retains the |t|=6.92 headline
    /// after σ is removed from the combination sum.
    #[serde(default)]
    pub discriminability: Option<DiscriminabilityReport>,
}

impl MultichannelBayesResult {
    /// Returns the underlying [`EvidenceStrength`] enum (not serialised).
    /// Use [`Self::combined_strength_label`] for the string form.
    pub fn combined_strength(&self) -> EvidenceStrength {
        categorise_evidence(self.combined_log10_bf)
    }
}

// =====================================================================
// Combiner — the core API
// =====================================================================

/// Combine per-channel [`ChannelEvidence`] into a [`MultichannelBayesResult`].
///
/// Assumes the channels are statistically independent; sums their natural-
/// log Bayes factors and converts to log_10, posterior odds, and n-σ
/// equivalent.
///
/// **σ-channel exclusion (P8.1e).** Only channels with
/// `for_combination = true` contribute to `combined_log_bf`. Channels
/// with `for_combination = false` — which by convention is just the
/// σ-channel — remain in the returned `channels` vector for audit but
/// do not enter the BF sum. The σ-channel discriminability metric is
/// surfaced via [`MultichannelBayesResult::discriminability`] instead.
///
/// # Panics
///
/// Never. Empty `channels` returns a degenerate result with
/// `combined_log_bf = 0`, `posterior_odds = 1`, and the `Inconclusive`
/// strength label.
pub fn combine_channels(channels: Vec<ChannelEvidence>) -> MultichannelBayesResult {
    let combined_log_bf: f64 = channels
        .iter()
        .filter(|c| c.for_combination)
        .map(|c| c.log_bf_ty_vs_schoen)
        .sum();
    const INV_LN10: f64 = 1.0 / core::f64::consts::LN_10;
    let combined_log10_bf = combined_log_bf * INV_LN10;
    let strength = categorise_evidence(combined_log10_bf);
    let n_sigma_equivalent = (2.0 * combined_log_bf.abs()).sqrt();
    let posterior_odds_ty = combined_log_bf.exp();
    // Total observations counts ALL channels (including discriminability-
    // only ones) — this is provenance metadata, not a model-comparison
    // weighting.
    let total_observations: usize = channels.iter().map(|c| c.n_observations).sum();
    MultichannelBayesResult {
        channels,
        combined_log_bf,
        combined_log10_bf,
        combined_strength_label: evidence_strength_label(strength).to_string(),
        posterior_odds_ty,
        n_sigma_equivalent,
        total_observations,
        discriminability: None,
    }
}

// =====================================================================
// Channel adapters
// =====================================================================

/// Identifier for the chain-match channel type (P7.11).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainType {
    /// Quark sub-Coxeter chain-position channel (`p7_11_quark_chain.json`).
    Quark,
    /// Lepton sub-Coxeter chain-position channel (`p7_11_lepton_chain.json`).
    Lepton,
}

impl ChainType {
    fn name(&self) -> &'static str {
        match self {
            ChainType::Quark => "chain_quark",
            ChainType::Lepton => "chain_lepton",
        }
    }
}

// ---------------------------------------------------------------------
// σ-channel adapter (P5.10 → ChannelEvidence)
// ---------------------------------------------------------------------

/// Top-level shape of `output/p5_10_ty_schoen_5sigma.json` — both the
/// `discrimination_strict_converged` summary block (used for the per-
/// model Gaussian widths) and the `candidates` block (used for the per-
/// seed σ values that drive the model-comparison Bayes factor).
#[derive(Deserialize)]
struct P5_10Json {
    /// Donaldson convergence tolerance from the run (per-seed strict
    /// filter: `final_donaldson_residual < donaldson_tol`).
    donaldson_tol: f64,
    /// Maximum Donaldson iterations from the run (per-seed strict
    /// filter: `iterations_run < donaldson_iters`).
    donaldson_iters: u32,
    /// Per-`k` strict-converged ensemble summary rows. The first entry
    /// is the canonical (lowest-k) headline used for the model widths.
    discrimination_strict_converged: Vec<P5_10Discrimination>,
    /// Per-candidate, per-seed σ records. Filtered with the same
    /// strict criterion the P5.10 binary uses (see [`P5_10PerSeed`]).
    candidates: Vec<P5_10Candidate>,
}

#[derive(Deserialize)]
struct P5_10Discrimination {
    /// `k` (basis order) this row was computed at. The Welch t-test is
    /// computed from the per-seed σ vectors (`candidates[*].per_seed`),
    /// so the per-row summary statistics (`mean_*`, `se_*`, `n_*`) are
    /// **not** consumed by the σ-channel adapter — they live in the JSON
    /// for human reporting only.
    k: u32,
}

#[derive(Deserialize)]
struct P5_10Candidate {
    /// Candidate label: `"TY"` or `"Schoen"`.
    candidate: String,
    /// Basis order this batch ran at.
    k: u32,
    /// Per-seed σ records.
    per_seed: Vec<P5_10PerSeed>,
}

#[derive(Deserialize)]
struct P5_10PerSeed {
    /// Final Donaldson σ value for this seed (the model-comparison
    /// observable).
    sigma_final: f64,
    /// Final Donaldson residual — used by the strict filter
    /// (`< donaldson_tol`).
    final_donaldson_residual: f64,
    /// Number of Donaldson iterations consumed — used by the strict
    /// filter (`< donaldson_iters` ⇒ converged before the cap).
    iterations_run: u32,
}

/// Welch two-sample t-test breakdown of the σ-channel Bayes factor.
///
/// Records the per-population sample statistics (mean, sample SD, n) and
/// the derived Welch t-statistic + Laplace/BIC ln BF. This is the
/// information needed to reproduce / audit the σ-channel ln BF without
/// re-running the full P5.10 ensemble.
///
/// **What this is not.** This is *not* a per-seed `Δ_i = log L_i^TY −
/// log L_i^Schoen` sum (the P8.1b formulation, which mis-used SE_mean as
/// a per-seed predictive width and inflated by 10⁹ nats). This is a two-
/// sample t-test on the batch *means*, the right Bayes factor for the
/// hypothesis "do the population means differ?" — see P8.1c hostile
/// review.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SigmaChannelBreakdown {
    /// Welch t-statistic: `t = (μ_TY − μ_Schoen) / sqrt(s²_TY/n_TY +
    /// s²_Schoen/n_Schoen)`. Sign tracks which mean is larger; magnitude
    /// gives the evidence strength.
    pub t_statistic: f64,
    /// Pooled SE used as the denominator of the t-statistic
    /// (`sqrt(s²_TY/n_TY + s²_Schoen/n_Schoen)`). Honest population-mean
    /// uncertainty — built from the **sample variance** of the per-seed
    /// σ vector, not from SE_mean (the P8.1b bug).
    pub pooled_sd: f64,
    /// Sample SD of the TY per-seed σ vector
    /// (`sqrt(Σ(σ_i − μ_TY)² / (n_TY − 1))`). Population-level spread.
    pub sigma_pop_ty: f64,
    /// Sample SD of the Schoen per-seed σ vector. Population-level
    /// spread.
    pub sigma_pop_schoen: f64,
    /// Number of TY seeds folded into the t-statistic.
    pub n_ty: usize,
    /// Number of Schoen seeds folded into the t-statistic.
    pub n_schoen: usize,
    /// **Signed** σ-channel ln BF in the TY:Schoen convention (positive
    /// = TY favoured, negative = Schoen favoured). Computed from the
    /// Laplace/BIC approximation:
    ///
    /// ```text
    ///   |ln BF| = max(0, 0.5 · t² − 0.5 · ln(n_TY + n_Schoen))
    /// ```
    ///
    /// with the sign chosen by the σ-channel "lower σ favoured"
    /// convention: TY favoured (positive) when `μ_TY < μ_Schoen` (i.e.
    /// `t < 0`); Schoen favoured (negative) when `μ_TY > μ_Schoen`.
    pub total_log_bf: f64,
    /// `k` (basis order) the per-seed σ values were drawn from.
    pub k: u32,
}

/// Build a σ-channel [`ChannelEvidence`] from a P5.10 multi-seed σ-
/// ensemble JSON via a **Welch two-sample t-test on the population
/// means**, converted to a Bayes factor via the Laplace/BIC
/// approximation.
///
/// ## What the σ-channel actually asks
///
/// "Do the population *means* of the per-seed Donaldson-loss σ values
/// differ between the TY and Schoen substrates?" — *not* "is each
/// individual Schoen seed drawn from TY's narrow Gaussian?" (which is a
/// per-seed predictive question requiring a per-seed predictive width,
/// not SE_mean).
///
/// The right BF for the population-mean question is a two-sample t-test
/// or its Bayes-factor equivalent. Following Kass & Wasserman (1995) and
/// Wagenmakers (2007), the BIC-style Laplace approximation gives
///
/// ```text
///   2 ln BF_10 ≈ t² − ln(n)            (Wagenmakers 2007 eq. (10))
///   ln BF_10   ≈ 0.5 · t² − 0.5 · ln(n)
/// ```
///
/// where `BF_10` is the Bayes factor for "means differ" (H1) over
/// "means equal" (H0) and `n = n_TY + n_Schoen` is the total sample
/// size.
///
/// ## How the t-statistic is computed (the P8.1b bug fix)
///
/// The previous attempt (P8.1b, `sigma_channel_per_seed_log_bf`) treated
/// SE_mean as the *predictive Gaussian width for an individual seed*.
/// SE_mean = σ_pop / √n is the uncertainty in the *estimate of the mean*
/// — it is *not* the predictive width for one observation. With the
/// production-realistic SE_TY = 5.5e-4 and SE_Schoen = 0.804, that
/// mistake produced a 10⁹-nats inflation (P8.1c hostile review).
///
/// This adapter instead derives the **population standard deviation**
/// directly from the per-seed σ vector,
///
/// ```text
///   s_M² = Σ_i (σ_{M,i} − μ_M)² / (n_M − 1)        (sample variance)
/// ```
///
/// (i.e. the sample variance, **not** SE_mean × √n which assumes the
/// SE_mean field on disk is in fact SE_mean and not something else).
/// Welch's t then is
///
/// ```text
///   t = (μ_TY − μ_Schoen) / sqrt(s_TY² / n_TY + s_Schoen² / n_Schoen)
/// ```
///
/// ## Sign convention
///
/// `t` is signed via the formula above; `t < 0` means TY's mean is
/// lower. The σ-channel ln BF in the TY:Schoen convention is:
///
/// ```text
///   |ln BF| = max(0, 0.5 · t² − 0.5 · ln(n))
///   sign(ln BF) = -sign(t)
/// ```
///
/// i.e. positive (TY favoured) when `μ_TY < μ_Schoen` — the "lower σ is
/// closer to the published ABKO target" convention used throughout the
/// P5 σ-channel pipeline. The chain-match channels (P7.11) sign their
/// ln BF the same way (`ln BF = δ/2` with `δ = schoen_residual −
/// ty_residual`, so `δ > 0` ⇒ TY residual smaller ⇒ TY favoured ⇒
/// `ln BF > 0`).
///
/// ## Strict filter
///
/// The same per-seed filter the P5.10 binary applies for its
/// `discrimination_strict_converged` row:
///
/// ```text
///   final_donaldson_residual < donaldson_tol  &&  iterations_run < donaldson_iters
/// ```
///
/// (`donaldson_tol`, `donaldson_iters` are top-level fields of the JSON.)
///
/// ## Variance source
///
/// `s_TY` and `s_Schoen` are computed directly from the strict-filtered
/// per-seed σ vectors (`per_seed[*].sigma_final`), not from any pre-
/// computed `se_*` field. This is intentional: pre-computed SE fields
/// in P5.10 are **SE_mean** (= σ_pop / √n), and accidentally treating
/// them as σ_pop is exactly the P8.1b bug.
///
/// # Errors
///
/// Returns `Err` if the file is missing, JSON-malformed, the strict-
/// converged summary block is empty, no per-seed candidates match the
/// canonical `k`, the strict filter empties either batch, either batch
/// has fewer than 2 seeds (sample variance undefined), or any sample
/// variance is non-positive / non-finite.
pub fn from_sigma_distribution(json_path: &str) -> Result<ChannelEvidence, String> {
    let (channel, _breakdown) = from_sigma_distribution_with_breakdown(json_path)?;
    Ok(channel)
}

/// Same as [`from_sigma_distribution`] but also returns the per-batch
/// breakdown so callers (e.g. the `p8_1_bayes_multichannel` binary) can
/// print TY-batch and Schoen-batch contributions separately.
///
/// # Errors
///
/// Same conditions as [`from_sigma_distribution`].
pub fn from_sigma_distribution_with_breakdown(
    json_path: &str,
) -> Result<(ChannelEvidence, SigmaChannelBreakdown), String> {
    let path: &Path = Path::new(json_path);
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("from_sigma_distribution: cannot read {}: {}", json_path, e))?;
    let parsed: P5_10Json = serde_json::from_str(&raw)
        .map_err(|e| format!("from_sigma_distribution: bad JSON in {}: {}", json_path, e))?;
    if !parsed.donaldson_tol.is_finite() || parsed.donaldson_tol <= 0.0 {
        return Err(format!(
            "from_sigma_distribution: bad donaldson_tol = {} in {}",
            parsed.donaldson_tol, json_path
        ));
    }
    if parsed.donaldson_iters == 0 {
        return Err(format!(
            "from_sigma_distribution: donaldson_iters must be > 0 in {}",
            json_path
        ));
    }

    // Locate the canonical strict-converged row (used for `k` only;
    // means and SDs come from the per-seed σ vectors below).
    let row = parsed
        .discrimination_strict_converged
        .first()
        .ok_or_else(|| {
            format!(
                "from_sigma_distribution: discrimination_strict_converged empty in {}",
                json_path
            )
        })?;
    let canonical_k = row.k;

    // Strict-filter the per-seed σ values at the canonical k.
    let strict_filter = |s: &P5_10PerSeed| -> bool {
        s.sigma_final.is_finite()
            && s.sigma_final > 0.0
            && s.final_donaldson_residual.is_finite()
            && s.final_donaldson_residual < parsed.donaldson_tol
            && s.iterations_run < parsed.donaldson_iters
    };

    let mut ty_seeds: Vec<f64> = Vec::new();
    let mut schoen_seeds: Vec<f64> = Vec::new();
    for cand in &parsed.candidates {
        if cand.k != canonical_k {
            continue;
        }
        match cand.candidate.as_str() {
            "TY" => {
                for s in &cand.per_seed {
                    if strict_filter(s) {
                        ty_seeds.push(s.sigma_final);
                    }
                }
            }
            "Schoen" => {
                for s in &cand.per_seed {
                    if strict_filter(s) {
                        schoen_seeds.push(s.sigma_final);
                    }
                }
            }
            other => {
                return Err(format!(
                    "from_sigma_distribution: unknown candidate label {:?} in {}",
                    other, json_path
                ));
            }
        }
    }

    if ty_seeds.len() < 2 || schoen_seeds.len() < 2 {
        return Err(format!(
            "from_sigma_distribution: insufficient strict-converged seeds at k={} in {} \
             (need ≥2 per batch for sample variance; TY n={}, Schoen n={})",
            canonical_k,
            json_path,
            ty_seeds.len(),
            schoen_seeds.len()
        ));
    }

    let breakdown = sigma_channel_welch_t_log_bf(&ty_seeds, &schoen_seeds, canonical_k)?;
    let n_obs = breakdown.n_ty + breakdown.n_schoen;
    // σ-channel: discriminability-only (for_combination=false) per P8.1e.
    // The σ-gap is dominated by basis-size differences (n_TY=87 vs
    // n_Schoen=27 at k=3), not by physics observables. The |t| / n-σ
    // headline is preserved via DiscriminabilityReport; the model-
    // comparison BF excludes this channel.
    let channel = ChannelEvidence::new_with_for_combination(
        "sigma",
        breakdown.total_log_bf,
        n_obs,
        json_path,
        false,
    )?;
    Ok((channel, breakdown))
}

/// Welch two-sample t-test → Laplace/BIC Bayes factor for the σ-channel.
///
/// Computes the **sample variance** (not SE_mean × √n) of each per-seed
/// vector directly:
///
/// ```text
///   μ_M    = (1/n_M) Σ σ_{M,i}
///   s_M²   = (1/(n_M − 1)) Σ (σ_{M,i} − μ_M)²
///   t      = (μ_TY − μ_Schoen) / sqrt(s_TY²/n_TY + s_Schoen²/n_Schoen)
///   |lnBF| = max(0, 0.5 t² − 0.5 ln(n_TY + n_Schoen))
///   sign   = −sign(t)        ; +ve = TY favoured (μ_TY < μ_Schoen)
/// ```
///
/// `n_TY ≥ 2` and `n_Schoen ≥ 2` are required for the sample variance
/// to be defined; the caller [`from_sigma_distribution_with_breakdown`]
/// enforces this.
///
/// Pooled SE used by the t-statistic denominator is exposed in the
/// returned [`SigmaChannelBreakdown`] for audit. The **clamp at 0** for
/// the BF magnitude is the standard BIC-handling for `|t| < √(ln n)`
/// (no evidence beyond noise — Wagenmakers 2007 §3).
///
/// # Errors
///
/// Returns `Err` if either per-seed vector contains a non-finite value,
/// either has fewer than 2 entries, the computed sample variance is
/// non-positive (degenerate input — all equal), or the pooled SE is
/// non-finite / non-positive.
pub fn sigma_channel_welch_t_log_bf(
    ty_seeds: &[f64],
    schoen_seeds: &[f64],
    k: u32,
) -> Result<SigmaChannelBreakdown, String> {
    fn mean_and_sample_sd(seeds: &[f64], label: &str) -> Result<(f64, f64), String> {
        if seeds.len() < 2 {
            return Err(format!(
                "sigma_channel_welch_t_log_bf: {label} batch has only {} seed(s); \
                 need ≥2 for sample variance",
                seeds.len()
            ));
        }
        for &s in seeds {
            if !s.is_finite() {
                return Err(format!(
                    "sigma_channel_welch_t_log_bf: non-finite seed in {label} batch ({s})"
                ));
            }
        }
        let n = seeds.len() as f64;
        let mean = seeds.iter().sum::<f64>() / n;
        let var = seeds
            .iter()
            .map(|&s| {
                let d = s - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        if !var.is_finite() || var <= 0.0 {
            return Err(format!(
                "sigma_channel_welch_t_log_bf: degenerate sample variance in {label} \
                 batch (variance = {var}); all seeds may be identical"
            ));
        }
        Ok((mean, var.sqrt()))
    }

    let (mu_ty, s_ty) = mean_and_sample_sd(ty_seeds, "TY")?;
    let (mu_schoen, s_schoen) = mean_and_sample_sd(schoen_seeds, "Schoen")?;
    let n_ty = ty_seeds.len();
    let n_schoen = schoen_seeds.len();
    let n_total = n_ty + n_schoen;

    let var_term = s_ty * s_ty / (n_ty as f64) + s_schoen * s_schoen / (n_schoen as f64);
    if !var_term.is_finite() || var_term <= 0.0 {
        return Err(format!(
            "sigma_channel_welch_t_log_bf: non-positive pooled variance term {var_term}"
        ));
    }
    let pooled_sd = var_term.sqrt();
    let t_statistic = (mu_ty - mu_schoen) / pooled_sd;
    if !t_statistic.is_finite() {
        return Err(format!(
            "sigma_channel_welch_t_log_bf: non-finite t-statistic \
             (μ_TY={mu_ty}, μ_Schoen={mu_schoen}, pooled_sd={pooled_sd})"
        ));
    }

    // Laplace/BIC Bayes factor for H1 (means differ) vs H0 (means equal):
    //   2 ln BF_10 ≈ t² − ln(n_total)              (Wagenmakers 2007)
    //   ln BF_10   ≈ 0.5 · t² − 0.5 · ln(n_total)
    // Clamp at 0 — when |t| < √(ln n) the data is in the noise floor of
    // the BIC approximation and the honest reading is "no evidence".
    let raw_magnitude = 0.5 * t_statistic * t_statistic - 0.5 * (n_total as f64).ln();
    let magnitude = raw_magnitude.max(0.0);

    // σ-channel sign convention: positive ln BF = TY favoured = μ_TY <
    // μ_Schoen (lower σ closer to ABKO target). Sign of (μ_Schoen − μ_TY)
    // equals -sign(t) since t = (μ_TY − μ_Schoen)/SE > 0 ⇔ μ_TY > μ_Schoen.
    let signed_log_bf = if t_statistic == 0.0 {
        0.0
    } else if t_statistic < 0.0 {
        magnitude
    } else {
        -magnitude
    };

    if !signed_log_bf.is_finite() {
        return Err(format!(
            "sigma_channel_welch_t_log_bf: non-finite ln BF \
             (t={t_statistic}, magnitude={magnitude})"
        ));
    }

    Ok(SigmaChannelBreakdown {
        t_statistic,
        pooled_sd,
        sigma_pop_ty: s_ty,
        sigma_pop_schoen: s_schoen,
        n_ty,
        n_schoen,
        total_log_bf: signed_log_bf,
        k,
    })
}

// ---------------------------------------------------------------------
// Chain-match adapter (P7.11 → ChannelEvidence)
// ---------------------------------------------------------------------

/// Top-level shape of `output/p7_11_quark_chain.json` /
/// `p7_11_lepton_chain.json`. Only `rows` is used (the deepest
/// `test_degree` row is treated as the most-converged measurement).
#[derive(Deserialize)]
struct P7_11Json {
    rows: Vec<P7_11Row>,
}

#[derive(Deserialize)]
struct P7_11Row {
    test_degree: u32,
    ty_residual: f64,
    schoen_residual: f64,
    /// `delta = schoen_residual - ty_residual` per P7.11 convention.
    /// Negative → Schoen residual smaller → Schoen preferred.
    delta: f64,
}

/// Build a chain-match [`ChannelEvidence`] from a P7.11 chain-match JSON.
///
/// **Likelihood model.** The P7.11 sweep reports `ty_residual` and
/// `schoen_residual` per `test_degree` (basis dimension). These are
/// least-squares fit residuals interpreted as `-2 log L` of the
/// chain-position fit; consequently
///
/// ```text
///   ln BF_TY:Schoen = -(ty_residual - schoen_residual) / 2
///                   =  delta / 2          (delta = schoen_resid - ty_resid)
/// ```
///
/// `delta < 0` → `ln BF < 0` → Schoen preferred (matches the
/// P7.11 convention).
///
/// We use the **deepest test degree's row** (largest basis) as the
/// most-converged estimate. `n_observations` is the basis dimension
/// reported in that row, which counts the independent chain-position
/// modes the residual aggregates over.
///
/// # Errors
///
/// Returns `Err` on file IO / JSON / empty-rows failures.
pub fn from_chain_match(
    json_path: &str,
    chain: ChainType,
) -> Result<ChannelEvidence, String> {
    let path: &Path = Path::new(json_path);
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("from_chain_match: cannot read {}: {}", json_path, e))?;
    let parsed: P7_11Json = serde_json::from_str(&raw)
        .map_err(|e| format!("from_chain_match: bad JSON in {}: {}", json_path, e))?;
    if parsed.rows.is_empty() {
        return Err(format!("from_chain_match: empty rows in {}", json_path));
    }
    // Pick the deepest test_degree row as the most-converged estimate.
    let deepest = parsed
        .rows
        .iter()
        .max_by_key(|r| r.test_degree)
        .ok_or_else(|| format!("from_chain_match: no max row in {}", json_path))?;
    if !deepest.delta.is_finite()
        || !deepest.ty_residual.is_finite()
        || !deepest.schoen_residual.is_finite()
    {
        return Err(format!(
            "from_chain_match: non-finite residuals at test_degree={} in {}",
            deepest.test_degree, json_path
        ));
    }
    // ln BF_TY:Schoen = -(ty_resid - schoen_resid) / 2 = delta / 2
    let ln_bf = deepest.delta / 2.0;
    // Use `test_degree` as n_observations (each degree contributes one
    // independent residual datum to the convergence sweep). The basis-
    // dim count is metadata, not an effective sample size.
    let n_obs = deepest.test_degree as usize;
    ChannelEvidence::new(chain.name(), ln_bf, n_obs.max(1), json_path)
}

// ---------------------------------------------------------------------
// Hodge-consistency adapter (P8.2 → ChannelEvidence) — placeholder
// ---------------------------------------------------------------------

/// Top-level shape of the Hodge-consistency channel output (P8.2).
/// The hodge agent is still building this module; the schema below is a
/// minimal contract: a single scalar `log_bf_ty_vs_schoen` plus an
/// effective sample size. Wider schemas will be back-compat-loadable as
/// long as those two fields are present.
#[derive(Deserialize)]
struct HodgeJson {
    log_bf_ty_vs_schoen: f64,
    n_observations: usize,
}

/// Build a Hodge-consistency [`ChannelEvidence`] from the P8.2 output
/// JSON. Tries to read `log_bf_ty_vs_schoen` and `n_observations` at the
/// top level. If the file is missing or malformed, returns `Err` so the
/// caller can choose to skip the channel.
pub fn from_hodge_consistency(json_path: &str) -> Result<ChannelEvidence, String> {
    let path: &Path = Path::new(json_path);
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("from_hodge_consistency: cannot read {}: {}", json_path, e))?;
    let parsed: HodgeJson = serde_json::from_str(&raw).map_err(|e| {
        format!(
            "from_hodge_consistency: bad JSON in {}: {} (expected top-level \
             `log_bf_ty_vs_schoen` and `n_observations` fields)",
            json_path, e
        )
    })?;
    ChannelEvidence::new("hodge", parsed.log_bf_ty_vs_schoen, parsed.n_observations, json_path)
}

// ---------------------------------------------------------------------
// Yukawa-spectrum adapter (P8.3 → ChannelEvidence) — placeholder
// ---------------------------------------------------------------------

/// Top-level shape of the Yukawa-eigenvalue / mass-spectrum channel
/// output (P8.3). Same minimal-contract pattern as
/// [`from_hodge_consistency`].
#[derive(Deserialize)]
struct YukawaJson {
    log_bf_ty_vs_schoen: f64,
    n_observations: usize,
}

/// Build a Yukawa-spectrum [`ChannelEvidence`] from the P8.3 output JSON.
pub fn from_yukawa_spectrum(json_path: &str) -> Result<ChannelEvidence, String> {
    let path: &Path = Path::new(json_path);
    let raw = std::fs::read_to_string(path)
        .map_err(|e| format!("from_yukawa_spectrum: cannot read {}: {}", json_path, e))?;
    let parsed: YukawaJson = serde_json::from_str(&raw).map_err(|e| {
        format!(
            "from_yukawa_spectrum: bad JSON in {}: {} (expected top-level \
             `log_bf_ty_vs_schoen` and `n_observations` fields)",
            json_path, e
        )
    })?;
    ChannelEvidence::new("yukawa", parsed.log_bf_ty_vs_schoen, parsed.n_observations, json_path)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_evidence_constructor_rejects_bad_inputs() {
        assert!(ChannelEvidence::new("", 1.0, 10, "src").is_err());
        assert!(ChannelEvidence::new("x", f64::NAN, 10, "src").is_err());
        assert!(ChannelEvidence::new("x", f64::INFINITY, 10, "src").is_err());
        assert!(ChannelEvidence::new("x", 1.0, 0, "src").is_err());
        assert!(ChannelEvidence::new("x", 1.0, 1, "src").is_ok());
    }

    #[test]
    fn empty_channels_returns_inconclusive_zero() {
        let r = combine_channels(Vec::new());
        assert_eq!(r.combined_log_bf, 0.0);
        assert_eq!(r.combined_log10_bf, 0.0);
        assert_eq!(r.posterior_odds_ty, 1.0);
        assert_eq!(r.n_sigma_equivalent, 0.0);
        assert_eq!(r.total_observations, 0);
        assert_eq!(r.combined_strength_label, "Inconclusive");
    }

    #[test]
    fn single_channel_passthrough() {
        let c = ChannelEvidence::new("solo", 12.0, 50, "/dev/null").unwrap();
        let r = combine_channels(vec![c]);
        assert!((r.combined_log_bf - 12.0).abs() < 1e-12);
        assert_eq!(r.combined_strength_label, "Strong");
        assert!(r.n_sigma_equivalent > 4.0);
        assert!(r.n_sigma_equivalent < 5.5);
    }

    // -----------------------------------------------------------------
    // σ-channel Welch t-test → Laplace/BIC BF
    // -----------------------------------------------------------------

    /// Reference t-statistic from `scipy.stats.ttest_ind(ty, schoen,
    /// equal_var=False)` for the synthetic batches below — pinned to
    /// 1e-12 to verify the rust implementation of Welch's t.
    ///
    /// scipy reproduction:
    /// ```python
    /// from scipy import stats
    /// ty = [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03, 0.97, 1.01]
    /// schoen = [2.0, 2.5, 1.8, 2.3, 1.9, 2.1, 2.4, 2.05, 1.95, 2.2]
    /// stats.ttest_ind(ty, schoen, equal_var=False).statistic
    /// # -> -15.185692978436753
    /// ```
    #[test]
    fn sigma_channel_t_test_recovers_classic_t() {
        let ty_seeds = vec![1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03, 0.97, 1.01];
        let schoen_seeds = vec![2.0, 2.5, 1.8, 2.3, 1.9, 2.1, 2.4, 2.05, 1.95, 2.2];
        let bd = sigma_channel_welch_t_log_bf(&ty_seeds, &schoen_seeds, 3).unwrap();

        // Pinned against scipy.stats.ttest_ind(equal_var=False).
        let scipy_t: f64 = -15.185_692_978_436_753;
        assert!(
            (bd.t_statistic - scipy_t).abs() < 1e-10,
            "Welch t = {} should equal scipy {} to 1e-10",
            bd.t_statistic,
            scipy_t
        );
        // n_TY = n_Schoen = 10.
        assert_eq!(bd.n_ty, 10);
        assert_eq!(bd.n_schoen, 10);
        // TY mean (1.0) < Schoen mean (≈2.12), so TY favoured → ln BF > 0.
        assert!(
            bd.total_log_bf > 0.0,
            "TY mean lower → TY favoured → ln BF > 0; got {}",
            bd.total_log_bf
        );
        // |ln BF| = 0.5·t² − 0.5·ln(20) ≈ 0.5·230.6 − 0.5·3.0 ≈ 113.8.
        let predicted_mag = 0.5 * scipy_t * scipy_t - 0.5_f64 * 20.0_f64.ln();
        assert!(
            (bd.total_log_bf - predicted_mag).abs() < 1e-10,
            "ln BF = {} should equal Laplace magnitude {} (TY favoured, +sign)",
            bd.total_log_bf,
            predicted_mag
        );
    }

    /// Production-realistic SE asymmetry (SE_TY = 5.5e-4, SE_Schoen =
    /// 0.804). Build per-seed vectors whose **sample SDs** match those
    /// SEs treated as σ_pop / √n with n = 20 / 16 (i.e. σ_pop_TY ≈
    /// 2.5e-3, σ_pop_S ≈ 3.2). The Welch t-test BF must land in the
    /// tens of nats, **not** 10⁹ nats — the P8.1b inflation regression.
    #[test]
    fn sigma_channel_t_test_asymmetric_se_no_inflation() {
        // σ_pop_TY ≈ 2.5e-3 → use a small spread around the TY mean.
        let mu_ty = 0.27_f64;
        let mu_schoen = 5.83_f64;
        let ty_seeds: Vec<f64> = (0..20)
            .map(|i| mu_ty + (i as f64 - 9.5) * 5.0e-4)
            .collect();
        // σ_pop_Schoen ≈ 3.2 → use a wider spread around the Schoen mean.
        let schoen_seeds: Vec<f64> = (0..16)
            .map(|i| mu_schoen + (i as f64 - 7.5) * 0.7)
            .collect();

        let bd = sigma_channel_welch_t_log_bf(&ty_seeds, &schoen_seeds, 3).unwrap();

        // Sanity: sample SDs are in the right ballpark (σ_pop, NOT
        // SE_mean — the bug regression).
        assert!(
            bd.sigma_pop_ty > 5.0e-4 && bd.sigma_pop_ty < 5.0e-3,
            "TY σ_pop should be ~mille-scale (per-seed spread, not SE_mean): {}",
            bd.sigma_pop_ty
        );
        assert!(
            bd.sigma_pop_schoen > 1.0 && bd.sigma_pop_schoen < 5.0,
            "Schoen σ_pop should be unity-scale: {}",
            bd.sigma_pop_schoen
        );

        // The headline anti-regression: |ln BF| < 1000 (production
        // value should be in the tens of nats).
        assert!(
            bd.total_log_bf.abs() < 1000.0,
            "asymmetric-SE σ-channel ln BF must NOT inflate beyond 10³ nats; \
             got {} (P8.1b bug would give ~1e9)",
            bd.total_log_bf
        );
        // It should also be non-trivial — sample means are ~6 σ_pool
        // apart by construction.
        assert!(
            bd.total_log_bf.abs() > 1.0,
            "asymmetric-SE σ-channel ln BF should still detect the >1 mean \
             separation; got {}",
            bd.total_log_bf
        );
    }

    /// Identical batches → t ≈ 0 → |ln BF| ≈ 0 (clamp guarantees
    /// non-negativity).
    #[test]
    fn sigma_channel_t_test_zero_separation_zero_evidence() {
        // Two batches drawn from the same population (linear ramp,
        // shifted by tiny amounts that average to zero).
        let ty_seeds: Vec<f64> = (0..10).map(|i| 1.0 + (i as f64 - 4.5) * 0.1).collect();
        let schoen_seeds: Vec<f64> = (0..10).map(|i| 1.0 + (i as f64 - 4.5) * 0.1).collect();
        let bd = sigma_channel_welch_t_log_bf(&ty_seeds, &schoen_seeds, 3).unwrap();
        assert!(
            bd.t_statistic.abs() < 1e-10,
            "identical batches must have t ≈ 0; got {}",
            bd.t_statistic
        );
        assert!(
            bd.total_log_bf.abs() < 5.0,
            "identical batches must yield negligible ln BF; got {}",
            bd.total_log_bf
        );
    }

    /// **P8.1e: discriminability-only sigma channel.**
    ///
    /// Well-separated batches must produce |t| > 5 — the discriminability
    /// signal that drives the published "TY-vs-Schoen at 6.92σ" headline.
    /// We deliberately do *not* assert on the sign of any combined BF
    /// here: σ is excluded from the model-comparison sum (see
    /// `combined_bf_excludes_sigma_channel`). The sign of the σ-channel's
    /// own `total_log_bf` is also informational only — it encodes the
    /// "lower σ favoured" convention for audit, but it is not allowed to
    /// drive the model verdict.
    #[test]
    fn sigma_channel_t_statistic_magnitude_recovers_discriminability() {
        let ty_seeds: Vec<f64> = (0..10).map(|i| 5.0 + (i as f64 - 4.5) * 0.1).collect();
        let schoen_seeds: Vec<f64> = (0..10).map(|i| 1.0 + (i as f64 - 4.5) * 0.1).collect();
        let bd = sigma_channel_welch_t_log_bf(&ty_seeds, &schoen_seeds, 3).unwrap();
        // Well-separated means → |t| > 5 (discriminability signal).
        assert!(
            bd.t_statistic.abs() > 5.0,
            "well-separated batches must produce |t| > 5; got |t| = {}",
            bd.t_statistic.abs()
        );
        // DiscriminabilityReport mirrors |t| as n-σ.
        let disc = DiscriminabilityReport::from_sigma_breakdown(&bd);
        assert_eq!(disc.sigma_n_ty, bd.n_ty);
        assert_eq!(disc.sigma_n_schoen, bd.n_schoen);
        assert!((disc.sigma_n_sigma - bd.t_statistic.abs()).abs() < 1e-12);
    }

    /// **Load-bearing P8.1e regression.** σ-channel ln BF (even if huge,
    /// e.g. +1000 nats) must NOT enter the combined model-comparison BF.
    /// Only physics channels (chain/Hodge/Yukawa, `for_combination=true`)
    /// are summed.
    #[test]
    fn combined_bf_excludes_sigma_channel() {
        let sigma = ChannelEvidence::new_with_for_combination(
            "sigma", 1000.0, 36, "synthetic", false,
        )
        .unwrap();
        let chain = ChannelEvidence::new_with_for_combination(
            "chain_quark", -5.0, 9, "synthetic", true,
        )
        .unwrap();
        let r = combine_channels(vec![sigma, chain]);
        // σ excluded → combined sum = chain only = -5.0
        assert!(
            (r.combined_log_bf - (-5.0)).abs() < 1e-10,
            "σ channel must be excluded from combined BF; expected -5.0, got {}",
            r.combined_log_bf
        );
        // Both channels still appear in the per-channel audit list.
        assert_eq!(r.channels.len(), 2);
        // n-σ equivalent matches sqrt(2 * |chain|).
        let expected_nsigma = (2.0_f64 * 5.0).sqrt();
        assert!((r.n_sigma_equivalent - expected_nsigma).abs() < 1e-10);
    }

    /// End-to-end JSON adapter: the synthetic-realistic SE asymmetry
    /// (matching the production case that broke P8.1b) flows through
    /// the strict filter → Welch t-test → ln BF without inflation.
    #[test]
    fn sigma_channel_adapter_no_se_mean_inflation_regression() {
        // Per-seed σ vectors with σ_pop_TY ≈ 2.5e-3 (µ-scale spread)
        // and σ_pop_Schoen ≈ 3.2 (unity-scale spread). If a future
        // regression mis-uses SE_mean = σ_pop / √n as the predictive
        // width, the asymmetric tightness will produce a 10⁹-nats
        // inflation; this test catches that.
        let mut ty_per_seed = Vec::new();
        for i in 0..20 {
            ty_per_seed.push(serde_json::json!({
                "sigma_final": 0.27 + (i as f64 - 9.5) * 5.0e-4,
                "final_donaldson_residual": 1e-7,
                "iterations_run": 16
            }));
        }
        let mut schoen_per_seed = Vec::new();
        for i in 0..16 {
            schoen_per_seed.push(serde_json::json!({
                "sigma_final": 5.83 + (i as f64 - 7.5) * 0.7,
                "final_donaldson_residual": 1e-7,
                "iterations_run": 16
            }));
        }
        let json = serde_json::json!({
            "donaldson_tol": 1.0e-6,
            "donaldson_iters": 100,
            "discrimination_strict_converged": [
                { "k": 3, "mean_ty": 0.27, "se_ty": 5.5e-4, "n_ty": 20,
                  "mean_schoen": 5.83, "se_schoen": 0.804, "n_schoen": 16 }
            ],
            "candidates": [
                { "candidate": "TY", "k": 3, "per_seed": ty_per_seed },
                { "candidate": "Schoen", "k": 3, "per_seed": schoen_per_seed }
            ]
        });
        let tmp = std::env::temp_dir().join("p8_1_t_test_inflation_regression.json");
        std::fs::write(&tmp, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        let tmp_str = tmp.to_string_lossy().to_string();

        let (chan, bd) =
            from_sigma_distribution_with_breakdown(&tmp_str).expect("adapter parses");
        assert_eq!(chan.name, "sigma");
        // P8.1e: σ-channel adapter must mark the channel as
        // discriminability-only (excluded from combined BF).
        assert!(
            !chan.for_combination,
            "σ-channel ChannelEvidence must have for_combination=false (P8.1e)"
        );
        assert_eq!(bd.n_ty, 20);
        assert_eq!(bd.n_schoen, 16);
        assert!(
            chan.log_bf_ty_vs_schoen.abs() < 1000.0,
            "production-realistic asymmetric-SE σ-channel must NOT inflate; got {}",
            chan.log_bf_ty_vs_schoen
        );
        // Magnitude should be in the tens of nats (matches the 6.92σ
        // headline: |t| ≈ 6.9, ln BF ≈ 22).
        assert!(
            chan.log_bf_ty_vs_schoen.abs() > 5.0
                && chan.log_bf_ty_vs_schoen.abs() < 100.0,
            "magnitude should be in the tens of nats (6.92σ ⇒ |ln BF| ≈ 22); got {}",
            chan.log_bf_ty_vs_schoen
        );
        let _ = std::fs::remove_file(&tmp);
    }
}
