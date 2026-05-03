//! Routes 1 and 2 from chapter 8 of the substrate-physics book.
//!
//! Sister module to [`crate::route34`] (which implements the Schoen-side
//! η-integral and Killing-vector / Arnold-ADE machinery for Routes 3
//! and 4). This module covers the **substrate-side reading** of two
//! computational routes that the mainstream string-theory phenomenology
//! pipeline does not have access to:
//!
//! ## Route 1 — Empirical observables as boundary conditions on the CY3 metric
//!
//! Mainstream CY3 metric computation (Donaldson 2009; Anderson-Karp-
//! Lukas-Palti 2010; Larfors-Schneider 2021) solves Yau's PDE under
//! Ricci-flatness alone. The substrate framework adds a richer set of
//! empirically-measured boundary constraints, all derivable from
//! substrate-specific commitments already in the book:
//!
//!   * Coulomb `1/r²` falloff (`hyp_substrate_mode_strain_tail_structure`)
//!     constrains the photon-mediator-mode zero-mode wavefunction's
//!     long-range behaviour on the CY3.
//!   * Weak-interaction range `ℏ / (M_W c) ≈ 10⁻¹⁸ m` constrains the
//!     CY3 metric in the W/Z-mediator-mode zero-mode region.
//!   * Strong-force confinement at `~ 1 fm` constrains the CY3 metric
//!     so the gluon-class cross-term content has zero long-range
//!     strain tail.
//!   * Polyhedral-resonance pattern admissibility
//!     (`hyp_substrate_polyhedral_resonance_pattern`) constrains the
//!     CY3's continuous-isometry / Killing-vector-field structure to
//!     be ADE-compatible (the same hypothesis the [`crate::route34`]
//!     Killing-vector solver also tests).
//!
//! Mainstream string-theory phenomenology does not use these as
//! boundary conditions because mainstream framing treats them as
//! outputs to be derived from the metric. The substrate framework
//! treats them as constraints on the metric to be derived from.
//!
//! [`route1`] exposes scalar penalty functions that take a candidate
//! metric and a discrete observable and return a `≥ 0` violation
//! score; the metric solver in [`crate::quintic`] can fold these into
//! its Adam objective as additional terms.
//!
//! ## Route 2 — Cross-term sign mechanism as Yukawa-coupling determinant
//!
//! The substrate framework commits
//! (`hyp_substrate_coupling_as_cross_term`,
//! `hyp_substrate_force_unification_via_cross-term_sign`) that the
//! **Yukawa coupling between two participating modes IS the
//! substrate's cross-term at their mode-overlap**, with cross-term
//! sign determined by relative substrate-amplitude phase-alignment.
//! Mainstream Yukawa computation requires triple overlap integrals on
//! the CY3 with the metric in hand
//! ([`crate::yukawa_overlap::compute_yukawa_spectrum`] is the
//! mainstream-style implementation). The substrate-physical reading
//! suggests Yukawa magnitudes might be computable directly from
//! empirically-measured force magnitudes (Coulomb constant, weak
//! coupling, `α_s`) by reading them as cross-term magnitudes for
//! specific phase-alignment configurations.
//!
//! [`route2`] implements this leading-order reading: for the
//! gauge-coupling Yukawas, the magnitude is identified with the
//! corresponding gauge coupling at the relevant scale. This bypasses
//! the full overlap-integral computation for the gauge-coupling
//! Yukawas. The matter-Yukawas (Higgs-fermion-fermion) follow from
//! the same phase-alignment structure once the gauge ones are fixed,
//! but the precise reduction requires bundle-Hermite-Einstein data
//! that is itself a deferred research item (P3 in the lib-level
//! preamble); the matter-Yukawa side of Route 2 is therefore a
//! placeholder that surfaces the *form* of the prediction without
//! claiming the bundle-dependent prefactor.
//!
//! Both submodules are independent of [`crate::route34`] (Schoen-side
//! η-integral + Killing/Arnold) and can be built / tested in
//! isolation. The discrimination pipeline
//! ([`crate::pipeline::compute_5sigma_score_for_candidate`]) folds
//! their χ² contributions into the per-candidate `total_loss` once
//! the calling code is wired (see task #58 — currently pending).

pub mod route1;
pub mod route2;

#[cfg(test)]
mod tests {
    /// Smoke-check: both submodules compile and their public APIs
    /// surface. Functional tests live in each submodule's own
    /// `#[cfg(test)]` block.
    #[test]
    fn module_loads() {
        let _ = super::route1::polyhedral_resonance_penalty;
        let _ = super::route2::predict_gauge_yukawa_from_alpha;
    }
}
