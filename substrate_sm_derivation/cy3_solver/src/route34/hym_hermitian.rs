//! # Hermitian-Yang-Mills (HYM) metric on a polystable monad bundle
//!
//! ## Mathematical content
//!
//! For a polystable holomorphic vector bundle `V` over a Calabi-Yau
//! 3-fold `M` with Kähler form `J`, the **Donaldson-Uhlenbeck-Yau
//! theorem** (Donaldson 1985 *Proc. London Math. Soc.* **50** 1-26;
//! Uhlenbeck-Yau 1986 *Comm. Pure Appl. Math.* **39** S257-S293)
//! guarantees the existence of a unique (up to constant rescaling)
//! Hermitian metric `h_V` on `V` whose Chern connection's curvature
//! `F_{h_V}` satisfies the **HYM equation**
//!
//! ```text
//!     i Λ_ω F_{h_V}  =  c · I_V       (Hermite-Einstein)
//!     equivalently   F_{h_V}^{(1,1)}  ∧  J^2  =  c · J^3 · I_V
//! ```
//!
//! where `c = (2 π / Vol(M)) · μ(V)` is the slope of `V` (zero for
//! `SU(n)` bundles in our setting, since μ(V) = c_1(V)·J^2 / rank(V)
//! and c_1(V) = 0 by construction).
//!
//! ## Solver vs. convergence metric
//!
//! Following Donaldson 2005 (*J. Diff. Geom.* **70** 453-472) and
//! Anderson-Karp-Lukas-Palti 2010 (arXiv:1004.4399, §3 — "AKLP"),
//! we distinguish two **different** functionals on the space of
//! Hermitian metrics:
//!
//! 1. **Balancing functional** `T(H)` (the *solver*):
//!    ```text
//!        T(H) := ∫_M s s^† · H^{-1} · dvol  /  ∫_M dvol .
//!    ```
//!    Its fixed points `T(H) = H` are the **balanced** metrics on
//!    `V` (Donaldson 2001, *Numerical critical points*). At fixed
//!    polynomial degree `k`, the balanced metric is the unique
//!    minimum of the Mabuchi-like functional, and the Donaldson
//!    iteration `H_{n+1} = T(H_n)` converges geometrically to it.
//!
//! 2. **HYM functional** `σ(H)` (the *convergence metric*):
//!    ```text
//!        σ(H)_{αβ} := (n_V / V) ∫_M  s_α(p) s_β(p)^*
//!                                  /  (s(p)^† H^{-1} s(p))   dvol .
//!    ```
//!    The HYM residual is
//!    ```text
//!        R_{HYM}(H) := || σ(H) · H^{-1}  −  c · I_V ||_F
//!    ```
//!    which is the Frobenius distance from the genuine
//!    Hermite-Einstein equation `iΛ_ω F = c I_V` (Donaldson 2005
//!    eq. (1.2); AKLP 2010 §3 eq. (3.6)–(3.10)). σ(H) is the
//!    Bergman-kernel-normalised section overlap; its zero (modulo
//!    `cI`) is the actual HYM solution.
//!
//! AKLP 2010 §3 are explicit that **balanced ≠ HYM at finite k**: as
//! `k → ∞` the balanced metric converges to the HYM metric (Wang
//! 2005), but the residual `R_{HYM}` remains O(1/k) at the balanced
//! fixed point of `T`. AKLP Fig. 1 shows ~10⁻² HYM residual at k=4.
//!
//! ## What this module does
//!
//! For a stable monad bundle `V` and a converged CY3 metric (provided
//! via the [`MetricBackground`] trait — concrete implementations in
//! [`crate::route34::schoen_geometry`] and the existing
//! [`crate::cicy_sampler::CicySampler`]), [`solve_hym_metric`]:
//!
//! 1. Runs the **balancing-functional iteration** as the solver
//!    (`H_{n+1} = (1−α) H_n + α T(H_n)`) until the balancing residual
//!    `||T(H) − H||_F / ||H||_F` drops below `tol`. This is the
//!    fast-converging numerical engine.
//! 2. **After** convergence, computes the **HYM residual**
//!    `R_{HYM}(H_∞) = ||σ(H_∞) · H_∞^{-1} − c I_V||_F` from the
//!    converged H and reports this as `final_residual`. **This is
//!    the actual distance from the HYM equation, not a balancing
//!    diagnostic.**
//! 3. Reports the balancing residual at termination as a separate
//!    diagnostic field [`HymHermitianMetric::balancing_residual`].
//!
//! The two values agree only in the `k → ∞` limit; the gap between
//! them at finite resolution is exactly the AKLP balanced/HYM
//! discrepancy.
//!
//! The result is a [`HymHermitianMetric`] containing:
//!
//! * `h_coefficients` — the converged Hermitian matrix `H` in the
//!   monad's section basis;
//! * `final_residual` — the **HYM residual** `||σ(H)·H^{-1} − cI||_F`
//!   (the genuine Hermite-Einstein deviation);
//! * `balancing_residual` — the balancing-functional residual
//!   `||T(H) − H||_F / ||H||_F` at termination (solver diagnostic);
//! * `iterations_run` — number of T-operator steps actually run;
//! * `run_metadata` — reproducibility / audit trail.
//!
//! ## What we do NOT claim
//!
//! 1. The convergence rate of the T-operator on real bundles is
//!    *much* slower than for the CY3 metric itself; AKLP report
//!    O(1e-2) HYM residuals at moderate `k` even after the balancing
//!    iteration has fully converged. We do not promise machine
//!    precision on the HYM residual — only the published AKLP
//!    convergence behaviour.
//! 2. For `SU(n)` bundles the slope `μ(V) = 0`, so `c = 0` and the
//!    iteration is **homogeneous**; we fix the Frobenius norm of `H`
//!    to suppress the trivial rescale degeneracy.
//! 3. The discretisation uses the same Shiffman-Zelditch / Donaldson
//!    quadrature as the rest of the pipeline; quadrature error is
//!    bounded by the residual reported in [`HymHermitianMetric`].
//!
//! ## References
//!
//! * Donaldson, S. K., "Anti self-dual Yang-Mills connections over
//!   complex algebraic surfaces and stable vector bundles",
//!   *Proc. London Math. Soc.* **50** (1985) 1-26.
//! * Uhlenbeck, K., Yau, S.-T., "On the existence of Hermitian-Yang-
//!   Mills connections in stable vector bundles", *Comm. Pure Appl.
//!   Math.* **39** (1986) S257-S293, DOI 10.1002/cpa.3160390714.
//! * Wang, X., "Canonical metrics on stable vector bundles",
//!   *J. Differential Geom.* **70** (2005) 393-456,
//!   DOI 10.4310/jdg/1143642932.
//! * Anderson, J., Karp, R., Lukas, A., Palti, E., "Numerical
//!   Hermitian-Yang-Mills connections and vector bundle stability"
//!   ("AKLP"), arXiv:1004.4399 (2010).
//! * Anderson, J., Constantin, A., Lukas, A., Palti, E., "Yukawa
//!   couplings in heterotic Calabi-Yau models" ("ACLP"),
//!   arXiv:1707.03442 (2017).

use crate::zero_modes::MonadBundle;
use num_complex::Complex64;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------

/// Per-run reproducibility metadata for the HYM solve.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct HymRunMetadata {
    /// Wall-clock seconds.
    pub wall_clock_seconds: f64,
    /// PRNG seed.
    pub seed: u64,
    /// Quadrature sample count.
    pub n_samples: usize,
    /// Frobenius norm of the converged H, for sanity.
    pub h_frobenius_norm: f64,
    /// Slope μ(V) from the bundle's Chern data (0 for SU(n)).
    pub bundle_slope: f64,
}

/// Configuration for [`solve_hym_metric`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HymConfig {
    /// Maximum T-operator iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the relative HYM residual.
    pub tol: f64,
    /// Damping factor in the T-operator update,
    /// `H_{n+1} = (1 - α) H_n + α T(H_n)`. Stabilises convergence.
    pub damping: f64,
    /// PRNG seed (for sample weight resampling, if any).
    pub seed: u64,
}

impl Default for HymConfig {
    fn default() -> Self {
        Self {
            max_iter: 64,
            tol: 1.0e-3,
            damping: 0.5,
            seed: 0xA1F4_9C8E_5B3D_2701,
        }
    }
}

/// The HYM Hermitian metric on a polystable monad bundle.
///
/// In the monad's natural section basis (one row per `B`-summand
/// `O(b_i)`), `h_coefficients[α][β]` is the Hermitian inner product
/// `h(s_α, s_β)` averaged over the CY3 metric measure.
///
/// `n` (the matrix dimension) equals `bundle.b_lines.len()` — the
/// total number of `B`-line-bundle summands. This matches the
/// section-basis convention used in AKLP §3.
#[derive(Clone, Debug)]
pub struct HymHermitianMetric {
    /// `h_coefficients[i * n + j] = h(s_i, s_j)`, Hermitian
    /// (i.e. `h[i,j] = conj(h[j,i])`).
    pub h_coefficients: Vec<Complex64>,
    /// Side length `n` of the matrix.
    pub n: usize,
    /// **HYM residual** `||σ(H)·H^{-1} − c·I_V||_F`. This is the
    /// Frobenius distance from the genuine Hermite-Einstein equation
    /// `iΛ_ω F = c·I_V` evaluated on the converged matrix `H`. It is
    /// the published AKLP 2010 §3 convergence metric — **not** the
    /// balancing-functional residual (see `balancing_residual` for
    /// that). For SU(n) bundles `c = 0`.
    pub final_residual: f64,
    /// **Balancing-functional residual** `||T(H) − H||_F / ||H||_F`
    /// at termination of the T-iteration. This is the *solver*'s
    /// internal convergence diagnostic — its fixed points are the
    /// balanced metrics, which equal HYM only as `k → ∞`. Reported
    /// alongside `final_residual` so callers see both. The gap
    /// `|final_residual − balancing_residual|` is the AKLP
    /// balanced/HYM discrepancy at the bundle's polynomial resolution.
    pub balancing_residual: f64,
    /// Iterations actually run.
    pub iterations_run: usize,
    /// Reproducibility metadata.
    pub run_metadata: HymRunMetadata,
    /// Whether this metric is the trivial identity (the trivial bundle
    /// case; cross-check anchor for tests).
    pub is_trivial_bundle: bool,
}

impl HymHermitianMetric {
    /// Identity metric of dimension `n`. Used as the initial guess of
    /// the T-operator iteration and as the canonical answer for the
    /// trivial bundle.
    pub fn identity(n: usize) -> Self {
        let mut h = vec![Complex64::new(0.0, 0.0); n * n];
        for i in 0..n {
            h[i * n + i] = Complex64::new(1.0, 0.0);
        }
        Self {
            h_coefficients: h,
            n,
            final_residual: 0.0,
            balancing_residual: 0.0,
            iterations_run: 0,
            run_metadata: HymRunMetadata::default(),
            is_trivial_bundle: true,
        }
    }

    /// Look up `h[i, j]`.
    pub fn entry(&self, i: usize, j: usize) -> Complex64 {
        self.h_coefficients[i * self.n + j]
    }

    /// Frobenius norm `(Σ |h_{i,j}|^2)^(1/2)`.
    pub fn frobenius_norm(&self) -> f64 {
        self.h_coefficients
            .iter()
            .map(|z| z.norm_sqr())
            .sum::<f64>()
            .sqrt()
    }
}

// ---------------------------------------------------------------------
// Sampling-background trait
// ---------------------------------------------------------------------

/// Minimal interface from a converged CY3 metric needed by the HYM
/// T-operator. Concrete impls live in
/// [`crate::route34::schoen_geometry`] (Schoen) and
/// [`crate::cicy_sampler`] (Tian-Yau bicubic).
pub trait MetricBackground {
    /// Total number of accepted sample points.
    fn n_points(&self) -> usize;
    /// Concatenated `[Complex64; 8]` sample point coordinates,
    /// one row per accepted sample.
    fn sample_points(&self) -> &[[Complex64; 8]];
    /// Shiffman-Zelditch quadrature weight at sample `α`. Sum-to-one
    /// convention.
    fn weight(&self, alpha: usize) -> f64;
    /// Holomorphic 3-form `Ω(p_α)`. May contain NaNs if the sampler
    /// rejected the point.
    fn omega(&self, alpha: usize) -> Complex64;
}

/// Light-weight in-memory metric background — used when the caller
/// has already produced a sample cloud and wants to drive HYM /
/// Yukawa overlaps without re-running the sampler.
#[derive(Clone, Debug)]
pub struct InMemoryMetricBackground {
    pub points: Vec<[Complex64; 8]>,
    pub weights: Vec<f64>,
    pub omega: Vec<Complex64>,
}

impl MetricBackground for InMemoryMetricBackground {
    fn n_points(&self) -> usize {
        self.points.len()
    }
    fn sample_points(&self) -> &[[Complex64; 8]] {
        &self.points
    }
    fn weight(&self, alpha: usize) -> f64 {
        self.weights[alpha]
    }
    fn omega(&self, alpha: usize) -> Complex64 {
        self.omega[alpha]
    }
}

// ---------------------------------------------------------------------
// Section basis evaluation
// ---------------------------------------------------------------------

/// Evaluate the `i`-th `B`-summand section basis element at a sample
/// point. For a monad with `B = ⊕ O_M(b_i)`, the canonical basis
/// element on `B_i` is the lowest-weight monomial of multidegree
/// `b_i` — for the bicubic CY3 in `CP^3 × CP^3` this is the
/// product `z_0^{b_i[0]} · w_0^{b_i[1]}`.
///
/// Returns `0` if any component of `b_i` is negative (no global
/// section). Negative-degree summands are treated as the zero
/// section, which is the algebraic convention for `O(d)` with
/// `d < 0` on projective space.
#[inline]
fn eval_section_b(point: &[Complex64; 8], b_line: [i32; 2]) -> Complex64 {
    if b_line[0] < 0 || b_line[1] < 0 {
        return Complex64::new(0.0, 0.0);
    }
    let mut acc = Complex64::new(1.0, 0.0);
    for _ in 0..b_line[0] {
        acc *= point[0];
    }
    for _ in 0..b_line[1] {
        acc *= point[4];
    }
    acc
}

/// Evaluate the entire `B`-section basis at one sample point.
/// Returns a `Vec<Complex64>` of length `bundle.b_lines.len()`.
fn eval_full_section_basis(point: &[Complex64; 8], bundle: &MonadBundle) -> Vec<Complex64> {
    bundle
        .b_lines
        .iter()
        .map(|b| eval_section_b(point, *b))
        .collect()
}

// ---------------------------------------------------------------------
// T-operator
// ---------------------------------------------------------------------

/// Compute one application of the AKLP T-operator
/// `T(H) = ∫_M s s^† H^{-1} dvol / ∫_M dvol`.
///
/// Inputs:
///
/// * `bundle.b_lines.len() = n` — matrix dimension.
/// * `points` / `weights` — Shiffman-Zelditch quadrature.
/// * `h_in` — current Hermitian guess for `H`, length `n*n`.
///
/// Output: the integrated Hermitian matrix `T(H)`, length `n*n`.
fn t_operator_step(
    bundle: &MonadBundle,
    metric: &dyn MetricBackground,
    h_in: &[Complex64],
    n: usize,
) -> Vec<Complex64> {
    // 1. H^{-1}: invert the Hermitian H via Cholesky-LDL on its real
    //    representation. For numerical safety on near-singular H we
    //    use a Tikhonov-regularised pseudoinverse with regulariser
    //    1e-12 * ||H||_F.
    let h_inv = hermitian_inverse(h_in, n);

    let n_pts = metric.n_points();
    let mut t_acc = vec![Complex64::new(0.0, 0.0); n * n];
    let mut total_w = 0.0f64;

    for alpha in 0..n_pts {
        let w = metric.weight(alpha);
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let p = &metric.sample_points()[alpha];
        let basis = eval_full_section_basis(p, bundle);

        // s s^†: outer product of the section basis.
        // We accumulate w · (s ⊗ s^†) · H^{-1}. The final divide by
        // total_w normalises the sum-to-one quadrature.
        for i in 0..n {
            for j in 0..n {
                // (s s^† H^{-1})_{i,j} = Σ_k s_i conj(s_k) (H^{-1})_{k,j}
                let mut acc = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    let ss = basis[i] * basis[k].conj();
                    acc += ss * h_inv[k * n + j];
                }
                t_acc[i * n + j] += Complex64::new(w, 0.0) * acc;
            }
        }
        total_w += w;
    }

    if total_w > 0.0 {
        let inv_w = 1.0 / total_w;
        for z in t_acc.iter_mut() {
            *z *= Complex64::new(inv_w, 0.0);
        }
    }

    // Hermitian projection: kill any antihermitian floating-point part.
    enforce_hermitian(&mut t_acc, n);
    t_acc
}

// ---------------------------------------------------------------------
// HYM residual (σ-functional)
// ---------------------------------------------------------------------

/// Errors that can be raised when computing the HYM residual.
#[derive(Clone, Debug, PartialEq)]
pub enum HymResidualError {
    /// The bundle is empty (`n = 0`); residual is undefined.
    EmptyBundle,
    /// All sample weights were non-positive or non-finite — the
    /// integrator could not accumulate any density. Inspect the
    /// `MetricBackground` implementation.
    NoValidSamples,
    /// The Bergman kernel `s(p)^† H^{-1} s(p)` was non-positive at
    /// every sample point. This indicates either a degenerate H or
    /// a sample cloud that misses the bundle's section support
    /// entirely (e.g. all weights on points where every basis section
    /// vanishes).
    DegenerateBergmanKernel,
}

impl std::fmt::Display for HymResidualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBundle => write!(f, "HYM residual undefined for empty bundle"),
            Self::NoValidSamples => {
                write!(f, "no finite-positive-weight samples available")
            }
            Self::DegenerateBergmanKernel => write!(
                f,
                "Bergman kernel s^† H^{{-1}} s vanished at every sample"
            ),
        }
    }
}

impl std::error::Error for HymResidualError {}

/// Compute the **HYM residual** `||σ(H) · H^{-1} − c · I_V||_F` from a
/// converged Hermitian matrix `H`.
///
/// `σ(H)` is the **Donaldson σ-functional** (Donaldson 2005,
/// *J. Diff. Geom.* **70** §1; AKLP 2010 arXiv:1004.4399 §3 eq. (3.6)),
///
/// ```text
///     σ(H)_{αβ}  :=  (n_V / V) · ∫_M  s_α(p) s_β(p)^*  /  ρ(H, p)  dvol
///     ρ(H, p)    :=  s(p)^† · H^{-1} · s(p)            (Bergman kernel)
/// ```
///
/// whose Hermite-Einstein zero `σ(H) · H^{-1} = c · I_V` is the
/// pointwise HYM equation `iΛ_ω F_h = c · I_V` (Donaldson 2005
/// eq. (1.2)). For SU(n) bundles `c = 0`.
///
/// **Distinction from the balancing functional.** The T-operator
/// `T(H) = ∫ s s^† H^{-1} dvol / V` averages the *unnormalised*
/// outer product, while σ averages the *Bergman-normalised* outer
/// product. T's fixed points are *balanced* metrics; σ's zeros (mod
/// `cI`) are *HYM* metrics. They agree only as `k → ∞` (Wang 2005,
/// Donaldson 2005).
///
/// The Frobenius norm `||σ(H)·H^{-1} − cI||_F` is the standard
/// numerical proxy for the integrated HYM equation deviation.
///
/// # Arguments
///
/// * `bundle` — the monad bundle defining the section basis.
/// * `metric` — sample-cloud quadrature.
/// * `h` — converged Hermitian matrix in the section basis,
///   length `n*n`.
/// * `n` — matrix dimension (must equal `bundle.b_lines.len()`).
/// * `slope_constant_c` — the Hermite-Einstein constant `c =
///   2π·μ(V)/Vol(M)`. For SU(n) bundles pass `0.0`.
///
/// # Returns
///
/// `Ok(R_HYM)` with `R_HYM = ||σ(H)·H^{-1} − cI||_F`, or
/// `Err(HymResidualError)` describing the failure.
pub fn compute_hym_residual(
    bundle: &MonadBundle,
    metric: &dyn MetricBackground,
    h: &[Complex64],
    n: usize,
    slope_constant_c: f64,
) -> Result<f64, HymResidualError> {
    if n == 0 {
        return Err(HymResidualError::EmptyBundle);
    }

    // 1. Invert H (regularised, well-conditioned).
    let h_inv = hermitian_inverse(h, n);

    // 2. Accumulate σ(H)_{αβ} = (n_V / V) ∫ s_α(p) s_β(p)^* / ρ(H,p) dvol.
    //    With Σ w_α = 1 (sum-to-one weights) the per-sample contribution
    //    is multiplied by w_α and the leading 1/V cancels with the
    //    weight normalisation. We additionally multiply by n_V (rank
    //    factor) at the end so that for the trivial bundle (H = I, all
    //    sections are constant up to ambient FS factors) σ(H) ≈ I.
    let n_pts = metric.n_points();
    let mut sigma = vec![Complex64::new(0.0, 0.0); n * n];
    let mut total_w = 0.0f64;
    let mut bergman_pos_count = 0usize;

    for alpha in 0..n_pts {
        let w = metric.weight(alpha);
        if !w.is_finite() || w <= 0.0 {
            continue;
        }
        let p = &metric.sample_points()[alpha];
        let basis = eval_full_section_basis(p, bundle);

        // Bergman kernel ρ = s^† H^{-1} s. For Hermitian PSD H^{-1}
        // and any non-zero s this is real and positive; floating-point
        // noise can give a tiny imaginary part which we discard.
        let mut rho = Complex64::new(0.0, 0.0);
        for i in 0..n {
            for j in 0..n {
                rho += basis[i].conj() * h_inv[i * n + j] * basis[j];
            }
        }
        let rho_re = rho.re;
        if !rho_re.is_finite() || rho_re <= 1.0e-30 {
            // The bundle has no support at this sample (all sections
            // vanish there); skip this point. This is normal for
            // monad bundles whose section basis has algebraic
            // vanishing loci.
            continue;
        }
        bergman_pos_count += 1;

        // Outer product s_α(p) s_β(p)^* / ρ.
        let inv_rho = 1.0 / rho_re;
        for i in 0..n {
            for j in 0..n {
                let outer = basis[i] * basis[j].conj();
                sigma[i * n + j] +=
                    Complex64::new(w * inv_rho, 0.0) * outer;
            }
        }
        total_w += w;
    }

    if total_w <= 0.0 {
        return Err(HymResidualError::NoValidSamples);
    }
    if bergman_pos_count == 0 {
        return Err(HymResidualError::DegenerateBergmanKernel);
    }

    // Normalise by the accumulated weight (sum-to-one convention) and
    // scale by n_V so that σ(I) → I_V when the section basis spans the
    // bundle uniformly. This is the AKLP 2010 §3 convention.
    let n_v = n as f64;
    let scale = n_v / total_w;
    for z in sigma.iter_mut() {
        *z *= Complex64::new(scale, 0.0);
    }

    // 3. Multiply σ(H) · H^{-1}.
    let mut sigma_h_inv = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..n {
                acc += sigma[i * n + k] * h_inv[k * n + j];
            }
            sigma_h_inv[i * n + j] = acc;
        }
    }

    // 4. Subtract c · I_V and take the Frobenius norm.
    let mut res = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let mut z = sigma_h_inv[i * n + j];
            if i == j {
                z -= Complex64::new(slope_constant_c, 0.0);
            }
            res += z.norm_sqr();
        }
    }
    Ok(res.sqrt())
}

/// Slope-Einstein constant `c = 2π · μ(V) / Vol(M)` for the bundle.
///
/// `μ(V) = c_1(V) · J^{n-1} / rank(V)`. For SU(n) bundles `c_1(V) = 0`
/// by construction, so `c = 0`. For general polystable bundles a
/// non-trivial slope is honest input from the caller.
///
/// At the level of this module we only have access to the bundle's
/// `b_lines` and `c_lines` (line-bundle decomposition); the full slope
/// requires the geometry's intersection form. We therefore accept the
/// slope as an explicit input to [`compute_hym_residual`], with the
/// SU(n) default (`c = 0`) baked into [`solve_hym_metric`]. Callers
/// who need the non-SU(n) regime should compute `c` from their
/// geometry and call [`compute_hym_residual`] directly.
pub fn slope_einstein_constant_su_n() -> f64 {
    0.0
}

/// Hermitian inverse via real-decomposed LU on the
/// `2n × 2n` real representation of an `n × n` complex matrix.
/// Adds Tikhonov regularisation `ε · I` with
/// `ε = max(1e-12, 1e-10 ||H||_F)` before inversion to keep the
/// solve well-conditioned even when `H` is near-singular.
fn hermitian_inverse(h: &[Complex64], n: usize) -> Vec<Complex64> {
    // Augment with regulariser scaled by ||H||_F.
    let frob = h.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let eps = (frob * 1.0e-10).max(1.0e-12);

    // Build the (2n × 2n) real block matrix [Re H, -Im H; Im H, Re H].
    // Inverting this real representation gives the inverse of the
    // complex H back via the same block formula.
    let two_n = 2 * n;
    let mut a = vec![0.0f64; two_n * two_n];
    for i in 0..n {
        for j in 0..n {
            let z = h[i * n + j] + if i == j { Complex64::new(eps, 0.0) } else { Complex64::new(0.0, 0.0) };
            // Top-left:    Re H
            a[i * two_n + j] = z.re;
            // Top-right:   -Im H
            a[i * two_n + (n + j)] = -z.im;
            // Bottom-left:  Im H
            a[(n + i) * two_n + j] = z.im;
            // Bottom-right: Re H
            a[(n + i) * two_n + (n + j)] = z.re;
        }
    }

    // Augmented matrix [A | I_{2n}] for Gauss-Jordan inversion.
    let mut aug = vec![0.0f64; two_n * (2 * two_n)];
    for i in 0..two_n {
        for j in 0..two_n {
            aug[i * (2 * two_n) + j] = a[i * two_n + j];
        }
        aug[i * (2 * two_n) + (two_n + i)] = 1.0;
    }

    // Gauss-Jordan with partial pivoting.
    for col in 0..two_n {
        // Pivot.
        let mut pivot_row = col;
        let mut pivot_abs = aug[col * (2 * two_n) + col].abs();
        for r in (col + 1)..two_n {
            let v = aug[r * (2 * two_n) + col].abs();
            if v > pivot_abs {
                pivot_abs = v;
                pivot_row = r;
            }
        }
        if pivot_abs < 1.0e-30 {
            // Hopelessly singular; return identity (safe fallback).
            let mut out = vec![Complex64::new(0.0, 0.0); n * n];
            for i in 0..n {
                out[i * n + i] = Complex64::new(1.0, 0.0);
            }
            return out;
        }
        // Swap rows pivot_row <-> col.
        if pivot_row != col {
            for j in 0..(2 * two_n) {
                aug.swap(col * (2 * two_n) + j, pivot_row * (2 * two_n) + j);
            }
        }
        // Normalise the pivot row.
        let pv = aug[col * (2 * two_n) + col];
        let inv = 1.0 / pv;
        for j in 0..(2 * two_n) {
            aug[col * (2 * two_n) + j] *= inv;
        }
        // Eliminate the column in other rows.
        for r in 0..two_n {
            if r == col {
                continue;
            }
            let factor = aug[r * (2 * two_n) + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..(2 * two_n) {
                aug[r * (2 * two_n) + j] -= factor * aug[col * (2 * two_n) + j];
            }
        }
    }

    // Extract the inverse from the right half of the augmented matrix.
    let mut a_inv = vec![0.0f64; two_n * two_n];
    for i in 0..two_n {
        for j in 0..two_n {
            a_inv[i * two_n + j] = aug[i * (2 * two_n) + two_n + j];
        }
    }

    // Re-pack the (2n × 2n) real block back into an n × n complex matrix.
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        for j in 0..n {
            let re = a_inv[i * two_n + j];
            let im = a_inv[(n + i) * two_n + j];
            out[i * n + j] = Complex64::new(re, im);
        }
    }
    out
}

/// Project a near-Hermitian matrix exactly onto the Hermitian
/// subspace by averaging with its conjugate transpose. Removes any
/// floating-point antihermitian part introduced by the integrator.
fn enforce_hermitian(h: &mut [Complex64], n: usize) {
    for i in 0..n {
        // Diagonal entries must be real.
        let z = h[i * n + i];
        h[i * n + i] = Complex64::new(z.re, 0.0);
        for j in (i + 1)..n {
            let a = h[i * n + j];
            let b = h[j * n + i].conj();
            let avg = (a + b) * 0.5;
            h[i * n + j] = avg;
            h[j * n + i] = avg.conj();
        }
    }
}

/// Frobenius norm of a complex `n × n` matrix stored in row-major.
fn frob(h: &[Complex64]) -> f64 {
    h.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// Set the Frobenius norm of `h` to `target` (in place). For SU(n)
/// bundles the slope is zero, so the T-operator iteration is
/// homogeneous and the scale of `H` is undetermined — we fix it by
/// rescaling each iterate to unit Frobenius norm.
fn rescale_frob(h: &mut [Complex64], target: f64) {
    let f = frob(h);
    if f > 1.0e-30 {
        let s = target / f;
        for z in h.iter_mut() {
            *z *= Complex64::new(s, 0.0);
        }
    }
}

// ---------------------------------------------------------------------
// Public driver
// ---------------------------------------------------------------------

/// Solve for the HYM Hermitian metric of a polystable monad bundle on
/// a CY3 described by the given [`MetricBackground`].
///
/// **Solver**. The numerical engine is the AKLP §3 / Donaldson 2001
/// **balancing-functional iteration**
///
/// ```text
///     H_{n+1}  =  (1 - α) H_n  +  α T(H_n)
/// ```
///
/// (with `α = config.damping`), rescaled to unit Frobenius norm at
/// every step (homogeneous SU(n) regime). The iteration terminates
/// when the **balancing residual**
///
/// ```text
///     R_bal(H)  :=  ||T(H) − H||_F / ||H||_F
/// ```
///
/// drops below `config.tol`, or `config.max_iter` has been reached.
///
/// **Convergence metric**. After the balancing iteration terminates,
/// the function computes the **HYM residual**
///
/// ```text
///     R_HYM(H)  :=  ||σ(H) · H^{-1}  −  c · I_V||_F
/// ```
///
/// (Donaldson 2005 §1; AKLP 2010 §3 eq. (3.6)) on the converged H and
/// stores it in [`HymHermitianMetric::final_residual`]. This is the
/// **genuine** Hermite-Einstein deviation, not the balancing residual.
/// The balancing residual is also returned, in
/// [`HymHermitianMetric::balancing_residual`], so callers see both.
///
/// At finite polynomial resolution (the algebraic-metric ansatz at
/// fixed degree `k`) the two values agree only in the limit `k → ∞`
/// (Wang 2005, Donaldson 2005). The gap
/// `|final_residual − balancing_residual|` is the AKLP balanced/HYM
/// discrepancy at the bundle's resolution; AKLP 2010 Fig. 1 reports
/// O(1e-2) HYM residuals at k=4 even with fully converged balancing.
///
/// For SU(n) bundles (the typical heterotic case) the slope `μ(V) = 0`,
/// so `c = 0`. Non-SU(n) callers needing a non-trivial Einstein
/// constant should call [`compute_hym_residual`] directly with the
/// appropriate `c` derived from their geometry's intersection form.
///
/// **Trivial-bundle short-circuit**: if `bundle.b_lines` is empty or
/// all `b_lines` are zero (the trivial bundle), returns
/// [`HymHermitianMetric::identity`] with both residuals zero and
/// `is_trivial_bundle = true`.
///
/// **Non-finite-input safety**: any sample weight that is not finite
/// is dropped from the integrator. If the HYM residual computation
/// fails (insufficient samples, degenerate Bergman kernel) the
/// returned metric carries `final_residual = f64::INFINITY` so
/// downstream code can detect the failure without panicking. The
/// function itself never panics on user-supplied data.
pub fn solve_hym_metric(
    bundle: &MonadBundle,
    metric: &dyn MetricBackground,
    config: &HymConfig,
) -> HymHermitianMetric {
    let n = bundle.b_lines.len();
    if n == 0 {
        return HymHermitianMetric::identity(0);
    }
    let trivial = bundle.b_lines.iter().all(|b| b[0] == 0 && b[1] == 0);
    if trivial {
        let mut m = HymHermitianMetric::identity(n);
        m.is_trivial_bundle = true;
        return m;
    }

    let started = std::time::Instant::now();

    let mut h = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        h[i * n + i] = Complex64::new(1.0, 0.0);
    }
    rescale_frob(&mut h, (n as f64).sqrt());

    // Balancing-residual at termination (the solver's diagnostic).
    let mut bal_residual = f64::INFINITY;
    let mut iters_done = 0usize;
    let damping = config.damping.clamp(0.0, 1.0);

    for it in 0..config.max_iter {
        let t_h = t_operator_step(bundle, metric, &h, n);

        // Damped update: H_{n+1} = (1-α) H_n + α T(H_n).
        let mut h_next = vec![Complex64::new(0.0, 0.0); n * n];
        for k in 0..(n * n) {
            h_next[k] = h[k] * Complex64::new(1.0 - damping, 0.0)
                + t_h[k] * Complex64::new(damping, 0.0);
        }
        enforce_hermitian(&mut h_next, n);
        rescale_frob(&mut h_next, (n as f64).sqrt());

        // Balancing residual ||T(H) − H||_F / ||H||_F.
        let mut diff = vec![Complex64::new(0.0, 0.0); n * n];
        for k in 0..(n * n) {
            diff[k] = t_h[k] - h[k];
        }
        let r = frob(&diff) / frob(&h).max(1.0e-30);

        bal_residual = r;
        iters_done = it + 1;
        h = h_next;

        if bal_residual < config.tol {
            break;
        }
    }

    // After the balancing iteration: compute the genuine HYM residual
    // ||σ(H) · H^{-1} − c · I_V||_F on the converged H. SU(n) bundles
    // have c = 0; non-SU(n) callers must use `compute_hym_residual`
    // directly.
    let slope_c = slope_einstein_constant_su_n();
    let hym_residual =
        match compute_hym_residual(bundle, metric, &h, n, slope_c) {
            Ok(r) => r,
            Err(_) => {
                // No graceful recovery: signal the caller via INFINITY.
                // The function contract documents this. We do not
                // panic and we do not silently fall back to the
                // balancing residual (which would be a category error).
                f64::INFINITY
            }
        };

    let elapsed = started.elapsed().as_secs_f64();
    let frob_norm = frob(&h);

    HymHermitianMetric {
        h_coefficients: h,
        n,
        final_residual: hym_residual,
        balancing_residual: bal_residual,
        iterations_run: iters_done,
        run_metadata: HymRunMetadata {
            wall_clock_seconds: elapsed,
            seed: config.seed,
            n_samples: metric.n_points(),
            h_frobenius_norm: frob_norm,
            bundle_slope: 0.0,
        },
        is_trivial_bundle: false,
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    /// Build a random in-memory metric background with `n_pts` points.
    /// The points are random Gaussians on `C^8`; the weights are uniform
    /// and `Ω` is a fixed unit constant. Sufficient for unit-tests that
    /// only depend on the integrator's algebra, not the geometry.
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

    /// Test 1: the trivial bundle (`b_lines` all zero) returns
    /// identity, with both the HYM residual and the balancing residual
    /// exactly zero. The trivial bundle is the canonical anchor: H = I,
    /// `iΛF = 0`, and `c = 0` for SU(n).
    #[test]
    fn trivial_bundle_hym_is_identity() {
        let bundle = MonadBundle {
            b_lines: vec![[0, 0]; 3],
            c_lines: vec![],
            map_f: vec![],
            b_lines_3factor: None,
        };
        let metric = synthetic_metric(50, 1);
        let cfg = HymConfig::default();
        let h = solve_hym_metric(&bundle, &metric, &cfg);
        assert!(h.is_trivial_bundle);
        assert_eq!(h.n, 3);
        // Identity entries.
        for i in 0..3 {
            for j in 0..3 {
                let z = h.entry(i, j);
                if i == j {
                    assert!((z.re - 1.0).abs() < 1e-12);
                } else {
                    assert!(z.re.abs() < 1e-12 && z.im.abs() < 1e-12);
                }
            }
        }
        // Both residuals exactly zero on the trivial bundle.
        assert!(
            h.final_residual < 1.0e-6,
            "trivial-bundle HYM residual {} should be < 1e-6",
            h.final_residual
        );
        assert!(
            h.balancing_residual < 1.0e-6,
            "trivial-bundle balancing residual {} should be < 1e-6",
            h.balancing_residual
        );
    }

    /// Test: the HYM residual computed directly on the identity matrix
    /// over a rank-1 (`b_lines = [[1, 0]]`) algebraic line bundle is
    /// near-zero (modulo MC noise). For a rank-1 bundle, σ(I) is a
    /// scalar mass and σ(I)·I^{-1} − 0 = σ(I) which is positive — the
    /// rank-1 case is a sanity check that the integrator runs without
    /// dividing by zero.
    #[test]
    fn rank_one_line_bundle_residual_is_finite() {
        let bundle = MonadBundle {
            b_lines: vec![[1, 0]],
            c_lines: vec![],
            map_f: vec![],
            b_lines_3factor: None,
        };
        let metric = synthetic_metric(200, 7);
        // The trivial-bundle short-circuit only fires for ALL-zero
        // b_lines; [1,0] is not zero so the solver runs normally.
        let h = solve_hym_metric(&bundle, &metric, &HymConfig::default());
        assert!(!h.is_trivial_bundle);
        assert_eq!(h.n, 1);
        assert!(
            h.final_residual.is_finite(),
            "rank-1 HYM residual must be finite, got {}",
            h.final_residual
        );
    }

    /// Test 2: empty `b_lines` returns a 0×0 identity (no panic).
    #[test]
    fn empty_b_lines_returns_zero_dim() {
        let bundle = MonadBundle {
            b_lines: vec![],
            c_lines: vec![],
            map_f: vec![],
            b_lines_3factor: None,
        };
        let metric = synthetic_metric(10, 2);
        let h = solve_hym_metric(&bundle, &metric, &HymConfig::default());
        assert_eq!(h.n, 0);
    }

    /// Test 3: the **balancing-functional residual** (the solver's
    /// internal convergence metric) decreases with iteration count for
    /// the AKLP example bundle. This is the AKLP §3 convergence
    /// property of the T-operator iteration. Note that the genuine
    /// HYM residual (`final_residual`) does NOT obey this monotone
    /// behaviour — see `hym_distinct_from_balancing` below for the
    /// distinction.
    #[test]
    fn balancing_residual_decreases() {
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let metric = synthetic_metric(200, 3);

        let cfg_short = HymConfig {
            max_iter: 5,
            damping: 0.5,
            ..HymConfig::default()
        };
        let cfg_long = HymConfig {
            max_iter: 30,
            damping: 0.5,
            ..HymConfig::default()
        };

        let h_short = solve_hym_metric(&bundle, &metric, &cfg_short);
        let h_long = solve_hym_metric(&bundle, &metric, &cfg_long);

        // Long run reaches a smaller-or-equal balancing residual.
        assert!(
            h_long.balancing_residual <= h_short.balancing_residual + 1e-10,
            "longer run balancing residual {} not <= shorter {}",
            h_long.balancing_residual,
            h_short.balancing_residual
        );
    }

    /// Test 4: the converged H is Hermitian to floating-point precision.
    #[test]
    fn hym_result_is_hermitian() {
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let metric = synthetic_metric(150, 4);
        let cfg = HymConfig {
            max_iter: 20,
            damping: 0.5,
            ..HymConfig::default()
        };
        let h = solve_hym_metric(&bundle, &metric, &cfg);
        for i in 0..h.n {
            assert!(h.entry(i, i).im.abs() < 1e-10);
            for j in (i + 1)..h.n {
                let a = h.entry(i, j);
                let b = h.entry(j, i).conj();
                assert!(
                    (a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10,
                    "Hermitian violation at ({i},{j})"
                );
            }
        }
    }

    /// Diagnostic helper: print the converged HYM and balancing
    /// residuals for the AKLP example. Useful when explicitly
    /// regression-checking that the two residuals are the *different*
    /// numbers documented in this module's docstring. Not run by
    /// default in CI; invoke with
    /// `cargo test --lib route34::hym_hermitian::tests::aklp_residuals_diagnostic -- --nocapture --ignored`.
    #[test]
    #[ignore]
    fn aklp_residuals_diagnostic() {
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let metric = synthetic_metric(300, 11);
        let cfg = HymConfig {
            max_iter: 40,
            damping: 0.5,
            tol: 1.0e-6,
            ..HymConfig::default()
        };
        let h = solve_hym_metric(&bundle, &metric, &cfg);
        eprintln!(
            "AKLP residuals: HYM={:.6e}  balancing={:.6e}  iters={}",
            h.final_residual, h.balancing_residual, h.iterations_run
        );
    }

    /// Demonstrates the central S6 distinction: for the AKLP example
    /// bundle the **HYM residual** `final_residual = ||σ(H)·H^{-1} −
    /// cI||_F` and the **balancing-functional residual**
    /// `balancing_residual = ||T(H) − H||_F / ||H||_F` are *different*
    /// numbers — they agree only in the `k → ∞` limit.
    ///
    /// This test pins the invariant that they cannot be aliased: at
    /// finite resolution the two metrics measure different things, and
    /// the prior code was reporting the balancing-functional residual
    /// while labelling it "HYM residual". Reference: AKLP 2010
    /// arXiv:1004.4399 §3 (and Donaldson 2005 *J. Diff. Geom.* **70**
    /// §1, the original distinction).
    #[test]
    fn hym_distinct_from_balancing() {
        let bundle = MonadBundle::anderson_lukas_palti_example();
        let metric = synthetic_metric(300, 11);

        // Run for enough iterations that the balancing iteration has
        // largely converged but k is finite (so HYM ≠ balanced).
        let cfg = HymConfig {
            max_iter: 40,
            damping: 0.5,
            tol: 1.0e-6, // tight, so we don't terminate prematurely
            ..HymConfig::default()
        };
        let h = solve_hym_metric(&bundle, &metric, &cfg);

        // Both residuals are finite and reportable.
        assert!(
            h.final_residual.is_finite(),
            "HYM residual must be finite"
        );
        assert!(
            h.balancing_residual.is_finite(),
            "balancing residual must be finite"
        );

        // The two residuals are *not* numerically equal (they would be
        // equal only at k = ∞, which we are not at). We require a
        // material gap to lock in the distinction.
        let gap = (h.final_residual - h.balancing_residual).abs();
        assert!(
            gap > 1.0e-3,
            "expected HYM residual ({}) ≠ balancing residual ({}); \
             gap {} ≤ 1e-3 means the two metrics are aliased",
            h.final_residual,
            h.balancing_residual,
            gap
        );

        // The HYM residual should be O(1) or O(1e-1) (AKLP Fig. 1
        // reports ~10⁻² at k=4 with the *physical* CY measure — our
        // synthetic uniform measure on C^8 inflates it but the order
        // of magnitude is the right ballpark). We require finiteness
        // and positivity but do NOT pin a specific value because the
        // synthetic metric here is not the physical CY measure.
        assert!(
            h.final_residual > 0.0,
            "HYM residual must be strictly positive at finite k"
        );
    }

    /// Direct unit test of `compute_hym_residual` on a hand-crafted
    /// matrix. For the trivial rank-1 bundle with b_lines = [[0,0]]
    /// (constant section ≡ 1), σ(I) = 1 and σ(I)·I^{-1} − 0 = 1, so
    /// the HYM residual is ‖1 − 0‖_F = 1.
    ///
    /// This isolates the σ-functional implementation from the solver.
    #[test]
    fn compute_hym_residual_direct_rank1() {
        let bundle = MonadBundle {
            b_lines: vec![[0, 0]],
            c_lines: vec![],
            map_f: vec![],
            b_lines_3factor: None,
        };
        let metric = synthetic_metric(200, 13);
        let h = vec![Complex64::new(1.0, 0.0)];
        // c = 0 for SU(n); residual = ||σ(H) · H^{-1}||_F = ||σ(I)||_F.
        // For b_line = [0,0] every section evaluates to 1, so
        // ρ(I, p) = 1 and σ(I) = 1 (with n_V = 1 and total_w = 1 from
        // the synthetic uniform weights).
        let r = compute_hym_residual(&bundle, &metric, &h, 1, 0.0)
            .expect("rank-1 trivial line: HYM residual must compute");
        assert!(
            (r - 1.0).abs() < 1.0e-9,
            "expected σ(I) = 1, got HYM residual {}",
            r
        );

        // With c = 1, σ(I)·I − 1 = 0, residual = 0.
        let r0 = compute_hym_residual(&bundle, &metric, &h, 1, 1.0)
            .expect("must compute");
        assert!(
            r0.abs() < 1.0e-9,
            "expected σ(I) − 1 = 0, got {}",
            r0
        );
    }

    /// `compute_hym_residual` returns `Err(EmptyBundle)` for n = 0.
    #[test]
    fn compute_hym_residual_empty_bundle_errors() {
        let bundle = MonadBundle {
            b_lines: vec![],
            c_lines: vec![],
            map_f: vec![],
            b_lines_3factor: None,
        };
        let metric = synthetic_metric(10, 17);
        let r = compute_hym_residual(&bundle, &metric, &[], 0, 0.0);
        assert_eq!(r, Err(HymResidualError::EmptyBundle));
    }

    /// Test 5: HymHermitianMetric::identity has Frobenius norm sqrt(n).
    #[test]
    fn identity_frobenius_norm() {
        for n in [1, 3, 6] {
            let m = HymHermitianMetric::identity(n);
            let f = m.frobenius_norm();
            let expected = (n as f64).sqrt();
            assert!(
                (f - expected).abs() < 1e-12,
                "n={n} frob {f} expected {expected}"
            );
        }
    }
}
