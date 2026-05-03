//! Linear-algebra primitives — migrated to `pwos-math`.
//!
//! This module is a thin re-export shim. The implementations live in
//! [`pwos_math::linalg`] so they can be reused across the GDS monorepo
//! (CY3 substrate-discrimination, Pathways/PWOS, FictionMaker, etc.).
//!
//! The public API is unchanged: callers still use
//! `crate::linalg::{gemm, lu_decompose, solve_lu, invert}` exactly as before.

pub use pwos_math::linalg::{gemm, invert, lu_decompose, solve_lu};
