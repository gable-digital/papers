//! **Diagnostic-only** thread-local truncation override for the TY and
//! Schoen section bases.
//!
//! This module exists exclusively to support the
//! `p_basis_convergence_diag` binary, which empirically tests the
//! "σ-channel discrimination is dominated by basis-size differences"
//! claim from P8.1e/P8.4 by running σ-eval at matched basis sizes
//! across TY and Schoen. The production solvers leave the override at
//! `None` and behave bit-identically to the pre-existing pipeline.
//!
//! ## Why a thread-local instead of a config field
//!
//! Adding a `Cargo.toml`-published config field to `TyMetricConfig` /
//! `SchoenMetricConfig` would require updating the ~20 in-tree literal
//! initialisers that don't yet use `..Default::default()`, including
//! tests. To keep the diagnostic addition surgical (1 line touched in
//! each metric file, in basis-construction territory only — never the
//! Donaldson inner sums P-REPRO-2-fix-BC is editing), we use a
//! thread-local override consulted at exactly one site in each solver
//! immediately AFTER `build_*_invariant_reduced_basis` returns and
//! BEFORE the workspace is allocated.
//!
//! Production code paths never set the override; the field is `None`
//! everywhere except inside the diagnostic binary, which sets it
//! explicitly for each (CY3, k, n_b, seed) tuple.
//!
//! ## Contract
//!
//! * `set_truncation(Some(n))` — apply truncation `basis.truncate(n)`
//!   on the next solve on this thread (and only this thread).
//! * `set_truncation(None)` — production default, no-op.
//! * `get_truncation()` — solver consults this once per solve. After
//!   consultation the value is returned by `apply_truncation_if_set`
//!   (it does NOT auto-clear, the caller may keep iterating with the
//!   same truncation).
//!
//! Callers from outside the diagnostic binary MUST NOT touch this.

use std::cell::Cell;

thread_local! {
    static TRUNCATE: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Set (or clear) the basis truncation for the current thread.
pub fn set_truncation(n: Option<usize>) {
    TRUNCATE.with(|c| c.set(n));
}

/// Read the current truncation setting on this thread.
pub fn get_truncation() -> Option<usize> {
    TRUNCATE.with(|c| c.get())
}

/// Helper: in-place truncation of a basis `Vec<T>` to the first `n`
/// elements, if a truncation override is set on this thread AND
/// `n < basis.len()` AND `n > 0`. No-op for production calls.
pub fn apply_truncation_if_set<T>(basis: &mut Vec<T>) {
    if let Some(n) = get_truncation() {
        if n > 0 && n < basis.len() {
            basis.truncate(n);
        }
    }
}

/// RAII guard that sets the override and restores the previous value
/// on drop. Use this to scope a truncation to a single solver call:
///
/// ```ignore
/// let _g = TruncationGuard::new(Some(27));
/// let r = solve_ty_metric(cfg)?;
/// // override automatically cleared on drop
/// ```
pub struct TruncationGuard {
    prev: Option<usize>,
}

impl TruncationGuard {
    pub fn new(n: Option<usize>) -> Self {
        let prev = get_truncation();
        set_truncation(n);
        Self { prev }
    }
}

impl Drop for TruncationGuard {
    fn drop(&mut self) {
        set_truncation(self.prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_default_is_none() {
        assert_eq!(get_truncation(), None);
    }

    #[test]
    fn guard_sets_and_restores() {
        assert_eq!(get_truncation(), None);
        {
            let _g = TruncationGuard::new(Some(5));
            assert_eq!(get_truncation(), Some(5));
        }
        assert_eq!(get_truncation(), None);
    }

    #[test]
    fn nested_guards_restore_correctly() {
        {
            let _g1 = TruncationGuard::new(Some(10));
            assert_eq!(get_truncation(), Some(10));
            {
                let _g2 = TruncationGuard::new(Some(20));
                assert_eq!(get_truncation(), Some(20));
            }
            assert_eq!(get_truncation(), Some(10));
        }
        assert_eq!(get_truncation(), None);
    }

    #[test]
    fn apply_truncates_when_set() {
        let mut v = vec![1, 2, 3, 4, 5];
        let _g = TruncationGuard::new(Some(3));
        apply_truncation_if_set(&mut v);
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn apply_noop_when_n_exceeds_len() {
        let mut v = vec![1, 2, 3];
        let _g = TruncationGuard::new(Some(10));
        apply_truncation_if_set(&mut v);
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn apply_noop_when_unset() {
        let mut v = vec![1, 2, 3, 4, 5];
        apply_truncation_if_set(&mut v);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }
}
