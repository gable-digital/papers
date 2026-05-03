//! Build script — emits build-time environment variables consumed by the
//! `route34::repro` reproducibility manifest.
//!
//! Specifically:
//!   * `TARGET`      — the host triple (e.g. `x86_64-pc-windows-msvc`).
//!   * `RUSTC_VERSION` — best-effort, captured by running `rustc --version`
//!     under the build script's `RUSTC` env var. If the call fails we fall
//!     back to "unknown" so the manifest still serialises.
//!
//! Both variables are emitted via `cargo:rustc-env=` so they are available
//! to the rest of the crate via `env!()` at compile time.

use std::process::Command;

fn main() {
    // TARGET is unconditionally provided by Cargo to build scripts.
    let target = std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=TARGET={}", target);

    // RUSTC is set by Cargo to the rustc binary used for this build. Asking
    // it for `--version` gives us the reproducible compiler identity.
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let rustc_version = Command::new(&rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=RUSTC_VERSION={}", rustc_version);

    // Don't rerun unless the build script itself changes; the captured
    // values are stable per (host triple, toolchain) pair and re-emitting
    // them every build wastes incremental cache.
    println!("cargo:rerun-if-changed=build.rs");
}
