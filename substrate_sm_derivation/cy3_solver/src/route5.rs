//! Route 5 from chapter 8 of the substrate-physics book: scalar
//! spectral index `n_s` from `E_8 × E_8` Coxeter geometry.
//!
//! ## The three-step derivation chain (chapter 8, §"A New
//! Discrimination Channel")
//!
//! Step A — _horizon spherical packing_: the parent-black-hole
//! horizon's `S²` carries a finite spectrum of independent angular
//! substrate-amplitude modes. Substrate's E-type ADE catastrophe +
//! heterotic `E₈ × E₈` topology forces the angular-mode cutoff to
//! be the surviving `E_8` Coxeter content per heterotic sector:
//!
//! ```text
//!     ℓ_max = h_visible-survived + h_E8 = 2 h_E8 = 60
//! ```
//!
//! at the inversion-boundary level (pre-Wilson-line breaking).
//!
//! Step B — _e-fold count from Coxeter_: information conservation
//! across the inversion boundary forces the daughter region's
//! initial expansion-front `S²` to admit the same angular-mode
//! budget as the parent horizon. One `ℓ`-shell activates per
//! e-fold of superluminal expansion; total e-fold count is therefore
//!
//! ```text
//!     N = ℓ_max = 2 h_E8 = 60
//! ```
//!
//! Step C — _scalar spectral index_: in the perfect-de-Sitter limit
//! (infinite `N`), the primordial scalar power spectrum is exactly
//! scale-invariant (`n_s = 1`, Harrison-Zel'dovich-Peebles). The
//! finite-bootstrap correction breaks scale invariance with two
//! independent slow-roll-like contributions, each of order `1/N`,
//! summing to `2/N`:
//!
//! ```text
//!     n_s = 1 − 2/N = 1 − 2/60 = 58/60 ≈ 0.96666…
//! ```
//!
//! ## Discrimination structure
//!
//! Per chapter 8 §"Discrimination Program Implications":
//!
//! * The leading-order `58/60` is **candidate-CY3-independent**: both
//!   Tian-Yau Z/3 and Schoen Z/3 × Z/3 inherit the same unbroken-
//!   visible-sector `h_E8 = 30` at the inversion-boundary level
//!   (Wilson-line breaking acts on the daughter Standard-Model
//!   content, not the parent-side pre-symmetry-breaking heterotic
//!   structure). The leading-order prediction is therefore a
//!   _framework-vs-mainstream_ test, not a TY-vs-Schoen discriminator.
//! * The **merger-class correction** `ΔN ~ 1` is candidate-specific:
//!   merger-class birth involves multiple parent-side `S²` horizons
//!   contributing to the daughter's initial state; the candidate-
//!   CY3's Killing-vector projection structure at the inversion
//!   boundary determines how the angular-mode budgets sum. Tian-Yau
//!   Z/3 and Schoen Z/3 × Z/3 have different Killing-vector
//!   projection patterns (the same structure that drives Route 3 /
//!   Route 4 discrimination), so they predict different merger-class
//!   shifts of `n_s` at sub-leading order.
//!
//! ## Empirical anchors
//!
//! * **Planck 2018**: `n_s = 0.9649 ± 0.0042` (Aghanim et al. 2020).
//!   Framework's `58/60 ≈ 0.9667` sits ~ 0.4σ above the central value
//!   — currently flagged "Suggestive" at chapter-8 level.
//! * **CMB-S4 forecast**: ~0.001 precision, two orders of magnitude
//!   tighter than Planck. At that precision the merger-class sub-
//!   leading correction `Δn_s ~ 0.001` is observable, and the
//!   TY-vs-Schoen discrimination signal is observable.

pub mod spectral_index;
