//! Reproducibility manifest + repro_log helpers shared across the
//! discriminator binaries (P5.9, P5.10, P7.1).
//!
//! The headline 6.921σ result emitted by P5.10 is technically reproducible
//! (deterministic seeded RNG, captured git SHA, input config in JSON), but
//! prior to the additions in this file the JSON only carried `git_revision`.
//! For publication-grade auditability we capture:
//!
//!   * Git revision (already present; passed through here for symmetry).
//!   * Rust toolchain string (`rustc --version`) — captured at build time
//!     via `build.rs` and embedded with `env!("RUSTC_VERSION")`.
//!   * Target triple (`env!("TARGET")` from `build.rs`).
//!   * CPU brand string is intentionally OMITTED (would require a new
//!     dependency); we report `cpu_features` instead, which is the
//!     reproducibility-relevant quantity (different SIMD code paths).
//!   * `cpu_features` — `is_x86_feature_detected!` for the SIMD families
//!     that pwos-math `kernels/` dispatches on.
//!   * Hostname — `COMPUTERNAME` on Windows, `HOSTNAME` on Linux/macOS.
//!   * UTC timestamp in RFC 3339 (handcrafted to avoid pulling chrono).
//!   * Full command-line (`std::env::args()`) so the exact invocation is
//!     captured.
//!   * Number of rayon threads (`rayon::current_num_threads()`).
//!
//! The structured event stream (P5.10's per-seed events + run_start /
//! run_end) is written to `<output>.replog` via `replog_writer::ReplogFile`.
//! Each event is serialised as canonical JSON, hashed via
//! `pwos_math::infra::repro_log::sha256`, and chained: the hash of event N
//! is `sha256(prev_chain_hash || event_json_bytes)`. This lets a verifier
//! re-run the binary, replay the event stream, and confirm the final chain
//! hash matches the value captured in the JSON output.
//!
//! `pwos_math::infra::repro_log::ReproLog` is also instantiated alongside
//! the structured stream — its `kernel_name: &'static str` field is too
//! restrictive for variable-shape solver events, so we use it as the
//! "kernel-call" companion log (one entry per Donaldson solve, hashed by
//! the seed and the σ values produced).

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use pwos_math::infra::repro_log::{sha256, ReproLog};

// ---------------------------------------------------------------------------
// ReproManifest
// ---------------------------------------------------------------------------

/// Reproducibility manifest captured at run start. Embedded in the JSON
/// output of every discriminator binary for publication-grade auditability.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReproManifest {
    /// `git rev-parse HEAD` output, or `None` if not in a git checkout.
    pub git_revision: Option<String>,
    /// `rustc --version` captured at build time by `build.rs`.
    pub rust_toolchain: String,
    /// Target triple (e.g. `x86_64-pc-windows-msvc`) from `build.rs`.
    pub target_triple: String,
    /// CPU SIMD feature flags detected at runtime that influence kernel
    /// dispatch in `pwos_math::kernels::*`. Determinism-relevant because
    /// AVX-512 and AVX2 paths can give bit-different results in some
    /// reductions.
    pub cpu_features: Vec<String>,
    /// `COMPUTERNAME` (Windows) / `HOSTNAME` (Unix), or `None` if unset.
    pub hostname: Option<String>,
    /// RFC 3339 UTC timestamp at run start.
    pub timestamp_utc: String,
    /// Full command line: `std::env::args().collect()`.
    pub command_line: Vec<String>,
    /// `rayon::current_num_threads()` captured once at run start.
    pub n_threads_rayon: usize,
}

/// Best-effort `git rev-parse HEAD`. Mirrors the local helper that already
/// exists in each discriminator binary.
fn git_revision() -> Option<String> {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
            } else {
                None
            }
        })
}

/// Detect the SIMD feature set used by pwos-math kernel dispatch. Stable
/// on x86_64; an empty vec on other architectures (the discriminator only
/// runs on x86_64 today).
fn detect_cpu_features() -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        for name in &[
            "sse2", "sse4.1", "sse4.2", "avx", "avx2", "fma", "avx512f",
            "avx512vl", "avx512bw", "avx512dq",
        ] {
            // The macro takes a literal so we have to spell out each one.
            let detected = match *name {
                "sse2" => std::is_x86_feature_detected!("sse2"),
                "sse4.1" => std::is_x86_feature_detected!("sse4.1"),
                "sse4.2" => std::is_x86_feature_detected!("sse4.2"),
                "avx" => std::is_x86_feature_detected!("avx"),
                "avx2" => std::is_x86_feature_detected!("avx2"),
                "fma" => std::is_x86_feature_detected!("fma"),
                "avx512f" => std::is_x86_feature_detected!("avx512f"),
                "avx512vl" => std::is_x86_feature_detected!("avx512vl"),
                "avx512bw" => std::is_x86_feature_detected!("avx512bw"),
                "avx512dq" => std::is_x86_feature_detected!("avx512dq"),
                _ => false,
            };
            if detected {
                out.push((*name).to_string());
            }
        }
    }
    out
}

/// Compose an RFC 3339 UTC timestamp without pulling chrono.
fn rfc3339_utc_now() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    // Civil-date conversion (Howard Hinnant's "civil_from_days" algorithm).
    let z = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400);
    let hh = secs_of_day / 3600;
    let mm = (secs_of_day / 60) % 60;
    let ss = secs_of_day % 60;
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp.wrapping_sub(9) };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hh, mm, ss
    )
}

/// Read hostname from the platform-appropriate env var.
fn hostname() -> Option<String> {
    std::env::var("COMPUTERNAME")
        .ok()
        .or_else(|| std::env::var("HOSTNAME").ok())
}

impl ReproManifest {
    /// Collect the full reproducibility manifest at run start.
    ///
    /// Cheap (a single `git rev-parse` subprocess + a few env lookups);
    /// safe to call from any binary's `main`.
    pub fn collect() -> Self {
        Self {
            git_revision: git_revision(),
            rust_toolchain: env!("RUSTC_VERSION").to_string(),
            target_triple: env!("TARGET").to_string(),
            cpu_features: detect_cpu_features(),
            hostname: hostname(),
            timestamp_utc: rfc3339_utc_now(),
            command_line: std::env::args().collect(),
            n_threads_rayon: rayon::current_num_threads(),
        }
    }
}

// ---------------------------------------------------------------------------
// Replog writer — chained-SHA event stream + ReproLog companion.
// ---------------------------------------------------------------------------

/// One structured event in the chained-SHA replay log. The exact JSON
/// representation IS the canonical hashed payload, so any field added to
/// this enum (or to the `ReproManifest`/`PerSeedEvent` payload) changes
/// the chain hash deterministically.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReplogEvent {
    /// Run start: full manifest + serialized CLI config + label.
    RunStart {
        binary: String,
        manifest: ReproManifest,
        config_json: serde_json::Value,
    },
    /// One Donaldson solve completed.
    PerSeed(PerSeedEvent),
    /// Run end: discrimination summary tier point + CI lower bound at each
    /// tier (or an arbitrary JSON value the caller chooses).
    RunEnd {
        summary: serde_json::Value,
        total_elapsed_s: f64,
    },
}

/// Per-seed Donaldson-solve event written to the chained replay log.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PerSeedEvent {
    pub seed: u64,
    pub candidate: String,
    pub k: u32,
    pub iters_run: usize,
    pub final_residual: f64,
    pub sigma_fs_identity: f64,
    pub sigma_final: f64,
    pub n_basis: usize,
    pub elapsed_ms: f64,
}

/// One record in the on-disk replog. Each event is paired with the SHA-256
/// chain hash AT THIS POSITION: `chain[i] = sha256(chain[i-1] || event[i])`
/// with `chain[-1] = sha256("")`.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ReplogRecord {
    pub seq: u64,
    pub chain_sha256_hex: String,
    pub event: ReplogEvent,
}

/// Wrapper around `pwos_math::infra::repro_log::ReproLog` plus our own
/// chained-SHA structured-event stream.
///
/// The `ReproLog` companion is fed one entry per Donaldson solve so the
/// pwos-math contract is exercised; the structured `events` Vec is what
/// actually gets serialised to disk because it can carry the full per-seed
/// payload.
pub struct ReplogWriter {
    pub events: Vec<ReplogRecord>,
    pub repro_log: ReproLog,
    chain_hash: [u8; 32],
    next_seq: u64,
}

impl ReplogWriter {
    /// Build an empty writer with `pwos_math::ReproLog` capacity for the
    /// expected event count + slack for the run_start / run_end frames.
    pub fn new(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity + 4),
            repro_log: ReproLog::with_capacity(capacity + 4),
            // Initial chain seed: sha256("") matches NIST FIPS 180-4 §B.0
            // and is verifiable independently.
            chain_hash: sha256(b""),
            next_seq: 0,
        }
    }

    /// Append an event. Hashes the canonical JSON, advances the chain.
    /// Panics only if `serde_json` fails to serialise — which would imply
    /// a non-finite f64 reaching the wire and indicate a logic bug worth
    /// catching loudly.
    pub fn push(&mut self, event: ReplogEvent) {
        let event_bytes = serde_json::to_vec(&event)
            .expect("ReplogEvent serialises (no non-finite f64s on the wire)");
        // chain[i] = sha256(chain[i-1] || event_bytes_i)
        let mut concat: Vec<u8> = Vec::with_capacity(32 + event_bytes.len());
        concat.extend_from_slice(&self.chain_hash);
        concat.extend_from_slice(&event_bytes);
        self.chain_hash = sha256(&concat);
        let chain_hex = hex_lower(&self.chain_hash);

        // Mirror into pwos_math::ReproLog so the kernel-call contract is
        // exercised. We only have one truly variable field per event —
        // the event's own bytes — so we use it as the input_sha and the
        // chain hash as the output_sha. n_floats_in/out are repurposed
        // as event-byte length (informational only).
        let input_sha = sha256(&event_bytes);
        let n_in = event_bytes.len().min(u32::MAX as usize) as u32;
        let entry = pwos_math::infra::repro_log::ReproEntry {
            kernel_name: "discriminator_event",
            seed: self.next_seq,
            input_sha,
            output_sha: self.chain_hash,
            n_floats_in: n_in,
            n_floats_out: 32,
            t_ns: 0,
        };
        self.repro_log.push(entry);

        self.events.push(ReplogRecord {
            seq: self.next_seq,
            chain_sha256_hex: chain_hex,
            event,
        });
        self.next_seq = self.next_seq.saturating_add(1);
    }

    /// Final SHA-256 chain hash in lowercase hex.
    pub fn final_chain_hex(&self) -> String {
        hex_lower(&self.chain_hash)
    }

    /// Persist the structured events as JSON Lines to `path`. Each line is
    /// one `ReplogRecord`; a verifier re-runs the binary, recomputes the
    /// chain, and checks that the final hash matches.
    ///
    /// Also drains the pwos-math `ReproLog` into a `.kernel.replog`
    /// sidecar (binary format documented in `repro_log::drain_to_bytes`).
    pub fn write_to_path<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> std::io::Result<()> {
        use std::io::Write;
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut f = std::fs::File::create(path)?;
        for rec in &self.events {
            let line = serde_json::to_string(rec).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::Other, e)
            })?;
            f.write_all(line.as_bytes())?;
            f.write_all(b"\n")?;
        }
        // Trailer: one final record with the chain hash but no event.
        // Lets a downstream verifier confirm completion without parsing
        // the last line specially.
        let trailer = serde_json::json!({
            "trailer": true,
            "n_events": self.events.len(),
            "final_chain_sha256_hex": hex_lower(&self.chain_hash),
        });
        let trailer_str = serde_json::to_string(&trailer).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e)
        })?;
        f.write_all(trailer_str.as_bytes())?;
        f.write_all(b"\n")?;

        // pwos-math kernel-call companion log.
        let kernel_path = path.with_extension("kernel.replog");
        let bytes = self.repro_log.drain_to_bytes();
        std::fs::write(kernel_path, bytes)?;

        Ok(())
    }
}

fn hex_lower(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_collects_without_panic() {
        let m = ReproManifest::collect();
        // RUSTC_VERSION is supplied by build.rs; if "unknown" we know the
        // build script ran but rustc was unreachable — still a valid run.
        assert!(!m.rust_toolchain.is_empty());
        assert!(!m.target_triple.is_empty());
        assert!(m.n_threads_rayon >= 1);
        // Timestamp is valid RFC 3339 length.
        assert_eq!(m.timestamp_utc.len(), 20);
        assert!(m.timestamp_utc.ends_with('Z'));
    }

    #[test]
    fn replog_chain_is_deterministic() {
        let mk = |seed: u64| -> ReplogEvent {
            ReplogEvent::PerSeed(PerSeedEvent {
                seed,
                candidate: "TY".to_string(),
                k: 3,
                iters_run: 17,
                final_residual: 1e-7,
                sigma_fs_identity: 0.123,
                sigma_final: 0.456,
                n_basis: 27,
                elapsed_ms: 1234.5,
            })
        };
        let mut a = ReplogWriter::new(8);
        a.push(mk(42));
        a.push(mk(99));
        let mut b = ReplogWriter::new(8);
        b.push(mk(42));
        b.push(mk(99));
        assert_eq!(a.final_chain_hex(), b.final_chain_hex());
        // Different events ⇒ different chain.
        let mut c = ReplogWriter::new(8);
        c.push(mk(99));
        c.push(mk(42));
        assert_ne!(a.final_chain_hex(), c.final_chain_hex());
    }
}
