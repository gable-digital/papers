//! P-INFRA Fix 3 regression test.
//!
//! The closest-to-ω_fix picker MUST never return `by_sigmoid` as the
//! chosen normalisation. The sigmoid `λ / (λ + λ_max)` saturates to
//! 0.5 when λ ≪ λ_max, and 0.5 is fortuitously close to
//! `123/248 = 0.4960`, producing false ~0.81% matches across
//! every cell of the P7.7 sweep.
//!
//! Post-fix the picker's scheme list excludes `by_sigmoid` entirely,
//! and the JSON output schemas no longer include sigmoid fields.
//! This test scans the live binaries' source for the offending
//! identifier in code-bearing positions (we filter out comment lines
//! since the `by_sigmoid` token survives in P-INFRA-Fix-3 historical
//! notes for traceability).

use std::fs;
use std::path::PathBuf;

fn read_bin_source(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("src");
    p.push("bin");
    p.push(name);
    fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", p.display()))
}

/// Strip Rust line comments (`// ...` until EOL). Conservative: we
/// only handle the `//` form because the P-INFRA Fix 3 notes use
/// that style. Block comments are not introduced by these binaries.
fn strip_line_comments(src: &str) -> String {
    src.lines()
        .map(|line| {
            // Find unquoted `//`. We don't have nested string-literal
            // edge cases in these binaries, so a simple scan is safe.
            let mut in_str = false;
            let bytes = line.as_bytes();
            for i in 0..bytes.len() {
                let b = bytes[i];
                if b == b'"' {
                    // Track quote toggling, ignoring escape because
                    // these files don't have `\"` in critical lines.
                    in_str = !in_str;
                }
                if !in_str
                    && i + 1 < bytes.len()
                    && bytes[i] == b'/'
                    && bytes[i + 1] == b'/'
                {
                    return line[..i].to_string();
                }
            }
            line.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn assert_no_code_sigmoid(name: &str) {
    let src = read_bin_source(name);
    let stripped = strip_line_comments(&src);
    assert!(
        !stripped.contains("by_sigmoid"),
        "{} must not reference `by_sigmoid` in code (P-INFRA Fix 3)",
        name
    );
}

#[test]
fn p7_7_binary_does_not_reference_by_sigmoid_in_code() {
    assert_no_code_sigmoid("p7_7_higher_k_omega_fix.rs");
}

#[test]
fn p7_6_binary_does_not_reference_by_sigmoid_in_code() {
    assert_no_code_sigmoid("p7_6_z3xz3_h4_omega_fix.rs");
}

#[test]
fn p7_3_binary_does_not_reference_by_sigmoid_in_code() {
    assert_no_code_sigmoid("p7_3_bundle_laplacian_omega_fix.rs");
}
