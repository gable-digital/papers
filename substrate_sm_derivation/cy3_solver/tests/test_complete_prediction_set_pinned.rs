//! Regression pin for the canonical end-to-end prediction set
//! (`p_complete_prediction_set`).
//!
//! Post Path A research closure (2026-05-06): the κ=0 frame-collapse
//! values are now the framework's canonical prediction at the rank-4
//! BHOP-2005 level — flagged `framework_prediction_falsified` against
//! PDG. Charged-lepton and neutrino slots are STRUCTURALLY ABSENT at
//! degree-1 H^1 of the rank-4 SU(4) shadow (was `deferred`). See
//! `references/p_dp9_w_invariant_proj_2026-05-06.md` (commit
//! `a4e6231e`) for the closure argument.
//!
//! Guards against:
//!   * silent loss of the FRAMEWORK_PREDICTION_FALSIFIED flag on the
//!     12 visible-sector Yukawa + CKM + Λ_cc rows (m_u..m_b, CKM
//!     Wolfenstein, Λ_cc),
//!   * silent loss of the STRUCTURALLY_ABSENT flag on the lepton +
//!     neutrino + PMNS rows (m_e, m_μ, m_τ, m_ν1..3, PMNS bundle),
//!   * change to the canonical 27-row visible-sector parameter count,
//!   * change to the 16-slot dark-sector count,
//!   * change to the 6 falsifiability signatures count,
//!   * silent breakage of the upstream provenance chain
//!     (catalogue → yukawa → assembled, catalogue → dark),
//!   * regressions in the canonical complete-set SHA-256 (any change to
//!     the section payloads must come with a deliberate pin update).
//!
//! The test reads the JSON shipped at
//! `output/p_complete_prediction_set.json`. Run
//! `cargo run --release --bin p_complete_prediction_set` first.

use serde_json::Value;
use std::fs;
use std::path::PathBuf;

/// Canonical SHA-256 of the section_a..section_e payload at the time
/// the pin was first taken. Update deliberately when any section
/// payload changes.
///
/// 2026-05-06 update: status vocabulary flipped to verdict (a)
/// post Path A closure. SHA recomputed.
const PINNED_CANONICAL_SHA: &str =
    "9a4a8765b9d1f3a7df1dbf5c06d3a89c53bbbcddc2bc27aa42d78637866e39e4";

/// Top-level status string (2026-05-06): renamed from
/// "complete_prediction_set_first_pass" to reflect Path A closure.
const PINNED_TOP_LEVEL_STATUS: &str =
    "framework_prediction_with_falsification_verdict_a";

const EXPECTED_SECTION_A_ROWS: usize = 27;
const EXPECTED_SECTION_A_PHYSICAL: u64 = 8;
const EXPECTED_SECTION_A_FALSIFIED: u64 = 12;
const EXPECTED_SECTION_A_ABSENT: u64 = 7;
const EXPECTED_DARK_SLOTS: u64 = 16;
const EXPECTED_FALSIFIABILITY: u64 = 6;

fn artefact_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("output");
    p.push("p_complete_prediction_set.json");
    p
}

fn load_artefact() -> Value {
    let path = artefact_path();
    let bytes = fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "p_complete_prediction_set.json not found at {} (run \
             `cargo run --release --bin p_complete_prediction_set` first): {}",
            path.display(),
            e
        )
    });
    serde_json::from_slice(&bytes).expect("artefact is valid JSON")
}

fn canonical_sha256(v: &Value) -> String {
    use sha2::{Digest, Sha256};
    fn canonicalize(v: &Value) -> Value {
        match v {
            Value::Object(m) => {
                let mut sorted = std::collections::BTreeMap::new();
                for (k, vv) in m {
                    sorted.insert(k.clone(), canonicalize(vv));
                }
                Value::Object(sorted.into_iter().collect())
            }
            Value::Array(a) => Value::Array(a.iter().map(canonicalize).collect()),
            other => other.clone(),
        }
    }
    let canonical = canonicalize(v);
    let bytes = serde_json::to_vec(&canonical).unwrap();
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    hex::encode(hasher.finalize())
}

#[test]
fn top_level_status_pins_path_a_verdict() {
    let d = load_artefact();
    assert_eq!(
        d["status"].as_str().unwrap(),
        PINNED_TOP_LEVEL_STATUS,
        "Top-level status MUST be the post-Path-A closure verdict \
         '{}' (renamed 2026-05-06 from \
         'complete_prediction_set_first_pass').",
        PINNED_TOP_LEVEL_STATUS,
    );
}

#[test]
fn section_a_has_exactly_27_rows() {
    let d = load_artefact();
    let rows = d["section_a_sm_parameters"].as_array().unwrap();
    assert_eq!(
        rows.len(),
        EXPECTED_SECTION_A_ROWS,
        "Section A must have exactly {} rows (canonical SM + cosmological table)",
        EXPECTED_SECTION_A_ROWS
    );
}

#[test]
fn section_a_index_monotonic() {
    let d = load_artefact();
    let rows = d["section_a_sm_parameters"].as_array().unwrap();
    for (i, r) in rows.iter().enumerate() {
        let idx = r["index"].as_u64().unwrap();
        assert_eq!(
            idx,
            (i + 1) as u64,
            "Section A row {} has non-monotonic index {}",
            i,
            idx
        );
    }
}

#[test]
fn yukawa_rows_are_framework_prediction_falsified() {
    let d = load_artefact();
    let rows = d["section_a_sm_parameters"].as_array().unwrap();
    // m_u, m_c, m_t, m_d, m_s, m_b → all six MUST be
    // framework_prediction_falsified post Path A closure.
    for sym in ["m_u", "m_c", "m_t", "m_d", "m_s", "m_b"] {
        let row = rows
            .iter()
            .find(|r| r["symbol"] == sym)
            .unwrap_or_else(|| panic!("Row '{}' missing", sym));
        assert_eq!(
            row["status"].as_str().unwrap(),
            "framework_prediction_falsified",
            "Row '{}' MUST be framework_prediction_falsified \
             (rank-4 BHOP κ=0 prediction in tension with PDG)",
            sym
        );
    }
    // CKM Wolfenstein + delta_CKM → all five framework_prediction_falsified.
    for sym in ["lambda", "A", "rho_bar", "eta_bar", "delta_CKM"] {
        let row = rows
            .iter()
            .find(|r| r["symbol"] == sym)
            .unwrap_or_else(|| panic!("Row '{}' missing", sym));
        assert_eq!(
            row["status"].as_str().unwrap(),
            "framework_prediction_falsified",
            "CKM '{}' MUST be framework_prediction_falsified \
             (CKM = identity at rank-4 κ=0 fixed point vs non-trivial PDG)",
            sym
        );
        assert_eq!(
            row["value"].as_f64().unwrap(),
            0.0,
            "CKM '{}' value MUST be 0 at the rank-4 κ=0 fixed point",
            sym
        );
    }
}

#[test]
fn lepton_neutrino_pmns_rows_are_structurally_absent() {
    let d = load_artefact();
    let rows = d["section_a_sm_parameters"].as_array().unwrap();
    for sym in ["m_e", "m_mu", "m_tau", "m_nu1", "m_nu2", "m_nu3"] {
        let row = rows
            .iter()
            .find(|r| r["symbol"] == sym)
            .unwrap_or_else(|| panic!("Row '{}' missing", sym));
        assert_eq!(
            row["status"].as_str().unwrap(),
            "structurally_absent",
            "Row '{}' MUST be structurally_absent (e^c / ν^c slot empty \
             at degree-1 H^1 of rank-4 SU(4) shadow — Path A closure)",
            sym
        );
        assert_eq!(
            row["value_string"].as_str().unwrap(),
            "UNAVAILABLE",
            "Row '{}' value_string MUST be 'UNAVAILABLE' (structurally_absent)",
            sym
        );
    }
    // PMNS bundle row.
    let pmns = rows
        .iter()
        .find(|r| r["symbol"].as_str().unwrap_or("").starts_with("(theta_12"))
        .expect("PMNS bundle row missing");
    assert_eq!(
        pmns["status"].as_str().unwrap(),
        "structurally_absent",
        "PMNS row MUST be structurally_absent (no Y_ν at this construction level)"
    );
}

#[test]
fn higgs_sector_rows_are_physical() {
    let d = load_artefact();
    let rows = d["section_a_sm_parameters"].as_array().unwrap();
    for sym in ["v", "m_H", "lambda_H", "mu^2", "G_N"] {
        let row = rows
            .iter()
            .find(|r| r["symbol"] == sym)
            .unwrap_or_else(|| panic!("Row '{}' missing", sym));
        assert_eq!(
            row["status"].as_str().unwrap(),
            "physical",
            "Row '{}' MUST be physical (Higgs / Newton's G are derivable structurally)",
            sym
        );
    }
}

#[test]
fn cosmological_constant_row_is_falsified() {
    let d = load_artefact();
    let rows = d["section_a_sm_parameters"].as_array().unwrap();
    let cc = rows
        .iter()
        .find(|r| r["symbol"] == "Lambda_cc")
        .expect("Lambda_cc row missing");
    assert_eq!(
        cc["status"].as_str().unwrap(),
        "framework_prediction_falsified",
        "Lambda_cc MUST be framework_prediction_falsified \
         (tree-level SM problem; Stage 7 loop corrections deferred — \
         independent of Path A)"
    );
}

#[test]
fn status_summary_counts_match_table() {
    let d = load_artefact();
    let s = &d["section_e_status_summary"];
    assert_eq!(
        s["total_visible_sector_rows"].as_u64().unwrap(),
        EXPECTED_SECTION_A_ROWS as u64,
        "Section E.total_visible_sector_rows mismatch"
    );
    assert_eq!(
        s["physical"].as_u64().unwrap(),
        EXPECTED_SECTION_A_PHYSICAL,
        "Section E.physical mismatch"
    );
    assert_eq!(
        s["framework_prediction_falsified"].as_u64().unwrap(),
        EXPECTED_SECTION_A_FALSIFIED,
        "Section E.framework_prediction_falsified mismatch"
    );
    assert_eq!(
        s["structurally_absent"].as_u64().unwrap(),
        EXPECTED_SECTION_A_ABSENT,
        "Section E.structurally_absent mismatch"
    );
    assert_eq!(
        s["dark_sector_slots_total"].as_u64().unwrap(),
        EXPECTED_DARK_SLOTS,
        "Section E.dark_sector_slots_total mismatch"
    );
    assert_eq!(
        s["falsifiability_signatures_total"].as_u64().unwrap(),
        EXPECTED_FALSIFIABILITY,
        "Section E.falsifiability_signatures_total mismatch"
    );
    // The visible-sector status counts must sum to the total.
    let sum = s["physical"].as_u64().unwrap()
        + s["framework_prediction_falsified"].as_u64().unwrap()
        + s["structurally_absent"].as_u64().unwrap()
        + s["partially_fitted"].as_u64().unwrap();
    assert_eq!(
        sum,
        EXPECTED_SECTION_A_ROWS as u64,
        "physical + framework_prediction_falsified + structurally_absent \
         + partially_fitted MUST sum to total Section A rows"
    );
}

#[test]
fn dark_sector_has_16_physical_slots() {
    let d = load_artefact();
    let b = &d["section_b_dark_sector_slots"];
    let n = b["total_dark_mass_slots"].as_u64().unwrap();
    assert_eq!(
        n, EXPECTED_DARK_SLOTS,
        "Dark sector MUST have exactly {} slots",
        EXPECTED_DARK_SLOTS
    );
    let kappa_flag = b["kappa_0_placeholder"].as_bool().unwrap();
    assert!(
        !kappa_flag,
        "Dark-sector kappa_0_placeholder MUST be false (closed-form geometric)"
    );
    let slots = b["chain_slots"].as_array().unwrap();
    assert_eq!(slots.len(), EXPECTED_DARK_SLOTS as usize);
    for (i, slot) in slots.iter().enumerate() {
        assert_eq!(
            slot["slot_idx"].as_u64().unwrap(),
            i as u64,
            "Dark slot {} has wrong slot_idx",
            i
        );
        let aff = slot["kappa_0_affected"].as_bool().unwrap();
        assert!(
            !aff,
            "Dark slot {} MUST have kappa_0_affected=false",
            i
        );
        let mass = slot["predicted_mass_gev"].as_f64().unwrap();
        assert!(
            mass.is_finite() && mass > 0.0,
            "Dark slot {} mass {} is not finite + positive",
            i,
            mass
        );
    }
}

#[test]
fn omega_dm_h2_matches_planck() {
    let d = load_artefact();
    let c = &d["section_c_cosmology"];
    let pred = c["omega_dm_h2_predicted"].as_f64().unwrap();
    let planck = c["omega_dm_h2_planck"].as_f64().unwrap();
    let rel = (pred - planck).abs() / planck;
    assert!(
        rel < 1.0e-6,
        "Omega_DM h^2 predicted {} vs Planck {} (rel diff {})",
        pred,
        planck,
        rel
    );
}

#[test]
fn falsifiability_has_6_signatures_with_bounded_verdicts() {
    let d = load_artefact();
    let sigs = d["section_d_falsifiability_signatures"].as_array().unwrap();
    assert_eq!(sigs.len(), EXPECTED_FALSIFIABILITY as usize);
    let allowed = ["PASS", "DISCOVERABLE_BY_2030", "FAIL_unless_linear"];
    for s in sigs {
        let v = s["verdict"].as_str().unwrap();
        assert!(
            allowed.contains(&v),
            "Falsifiability verdict '{}' is outside the allowed set {:?}",
            v,
            allowed
        );
    }
}

#[test]
fn provenance_chain_is_internally_consistent() {
    let d = load_artefact();
    let p = &d["provenance_chain"];

    // Catalogue → Yukawa.
    let cat_sha = p["catalogue"]["replog_final_chain_sha256"]
        .as_str()
        .unwrap();
    let yk_match = p["yukawa_matrices"]["catalogue_replog_chain_sha256_match"]
        .as_str()
        .unwrap();
    assert_eq!(
        cat_sha, yk_match,
        "catalogue replog SHA must match yukawa.catalogue_replog_chain_sha256"
    );

    // Yukawa → Assembled.
    let yk_sha = p["yukawa_matrices"]["replog_final_chain_sha256"]
        .as_str()
        .unwrap();
    let asm_match = p["assembled_lagrangian"]["yukawa_replog_chain_sha256_match"]
        .as_str()
        .unwrap();
    assert_eq!(
        yk_sha, asm_match,
        "yukawa replog SHA must match assembled.provenance.yukawa_replog_chain_sha256"
    );

    // Catalogue → Dark.
    let dark_match = p["dark_sector"]["upstream_catalogue_replog_chain_sha_match"]
        .as_str()
        .unwrap();
    assert_eq!(
        cat_sha, dark_match,
        "catalogue replog SHA must match dark.upstream_catalogue_replog_chain_sha"
    );

    // Honesty invariants.
    let asm_status = p["assembled_lagrangian"]["status"].as_str().unwrap();
    assert_eq!(
        asm_status, "kappa_0_placeholder_lagrangian",
        "assembled.status MUST be 'kappa_0_placeholder_lagrangian' until Ext^1 lands"
    );
    let dark_k0 = p["dark_sector"]["kappa_0_placeholder"].as_bool().unwrap();
    assert!(
        !dark_k0,
        "dark.kappa_0_placeholder MUST be false (geometric, not Yukawa-derived)"
    );

    // Every chain SHA must be 64-char hex.
    for path in [
        ("catalogue", "replog_final_chain_sha256"),
        ("yukawa_matrices", "replog_final_chain_sha256"),
        ("assembled_lagrangian", "replog_final_chain_sha256"),
        ("dark_sector", "replog_final_chain_sha256"),
    ] {
        let sha = p[path.0][path.1].as_str().unwrap();
        assert_eq!(
            sha.len(),
            64,
            "provenance.{}.{} must be 64-char hex; got {}",
            path.0,
            path.1,
            sha
        );
        assert!(
            sha.chars().all(|c| c.is_ascii_hexdigit()),
            "provenance.{}.{} has non-hex characters",
            path.0,
            path.1
        );
    }
}

#[test]
fn canonical_complete_set_sha_pinned() {
    let d = load_artefact();
    let payload = serde_json::json!({
        "section_a_sm_parameters":             d["section_a_sm_parameters"],
        "section_b_dark_sector_slots":         d["section_b_dark_sector_slots"],
        "section_c_cosmology":                 d["section_c_cosmology"],
        "section_d_falsifiability_signatures": d["section_d_falsifiability_signatures"],
        "section_e_status_summary":            d["section_e_status_summary"],
    });
    let computed = canonical_sha256(&payload);
    let stored = d["canonical_complete_set_sha256"].as_str().unwrap();
    assert_eq!(
        computed, stored,
        "Computed canonical SHA does not match stored canonical SHA. \
         The artefact's section payloads have been mutated post-emission."
    );
    assert_eq!(
        computed, PINNED_CANONICAL_SHA,
        "Canonical complete-set SHA-256 changed.\n  expected: {}\n  computed: {}\n\
         If this change is intentional (e.g. EXT1-ENGAGEMENT landed and \
         visible-sector Yukawa rows were upgraded to physical), update \
         PINNED_CANONICAL_SHA in this test and explain in the commit.",
        PINNED_CANONICAL_SHA, computed
    );
}

#[test]
fn yukawa_rows_have_correct_kappa_0_values() {
    // Defence-in-depth: pin the actual κ=0 values so an accidental swap
    // of arr[0] vs arr[2] (which the user's spec calls out explicitly)
    // is caught.
    let d = load_artefact();
    let rows = d["section_a_sm_parameters"].as_array().unwrap();
    let m_u = rows.iter().find(|r| r["symbol"] == "m_u").unwrap();
    let m_t = rows.iter().find(|r| r["symbol"] == "m_t").unwrap();
    let m_c = rows.iter().find(|r| r["symbol"] == "m_c").unwrap();
    let m_u_val = m_u["value"].as_f64().unwrap();
    let m_c_val = m_c["value"].as_f64().unwrap();
    let m_t_val = m_t["value"].as_f64().unwrap();
    // Per the canonical-table spec: m_u=1668, m_c=1603, m_t=1556 (descending
    // is by raw eigenvalue magnitude, not PDG hierarchy — that's the κ=0 signature).
    assert!(
        (m_u_val - 1668.0171).abs() < 0.01,
        "m_u κ=0 value = {} (expected ~1668.017)",
        m_u_val
    );
    assert!(
        (m_c_val - 1603.1897).abs() < 0.01,
        "m_c κ=0 value = {} (expected ~1603.190)",
        m_c_val
    );
    assert!(
        (m_t_val - 1556.0536).abs() < 0.01,
        "m_t κ=0 value = {} (expected ~1556.054)",
        m_t_val
    );
}
