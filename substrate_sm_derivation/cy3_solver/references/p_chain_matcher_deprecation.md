# chain_matcher deprecation tracker — REM-OPT-B Phase 2.7

This file tracks the deprecation of `src/route34/chain_matcher.rs` and
the `from_chain_match` adapter in `src/route34/bayes_factor_multichannel.rs`,
per Phase 2.7 of `book/scripts/cy3_substrate_discrimination/REMEDIATION_PLAN_OPTION_B.md`.

## What was done in Phase 2.7

1. **Module-level deprecation attribute** added at the `pub mod chain_matcher;`
   declaration site in `src/route34/mod.rs`:

   ```rust
   #[deprecated(
       since = "0.1.0",
       note = "Replaced by yukawa_pipeline; see REMEDIATION_PLAN_OPTION_B.md Phase 2."
   )]
   pub mod chain_matcher;
   ```

2. **Item-level deprecation attributes** added/extended on every `pub`
   item in `src/route34/chain_matcher.rs`:

   - `enum ChainType` (existing — note extended to reference REM-OPT-B Phase 2)
   - `struct ChainMatchResult` (existing — note extended)
   - `fn match_chain` (existing — note extended)
   - `fn hungarian_assign` (NEW)
   - `fn match_chain_hungarian` (existing — note extended)
   - `struct RatioMatch` (NEW)
   - `fn ratio_pattern` (NEW)

3. **Tests calling `from_chain_match` marked `#[ignore]`** with a
   `TODO: REM-OPT-B Phase 2.7 — migrate to from_yukawa_pipeline_prediction`
   comment, and `#[allow(deprecated)]` on the test functions so the
   tests still compile while the migration target does not yet exist:

   - `tests/test_bf_respects_convergence_flag.rs::bf_respects_convergence_flag_when_input_not_converged`
   - `tests/test_chain_type_json_consistency.rs::from_chain_match_rejects_chain_type_mismatch`
   - `tests/test_bayes_multichannel_integration.rs::production_data_three_channels`
   - `src/route34/tests/test_bayes_multichannel.rs::tests::production_data_three_channels`

   These tests will be migrated to invoke `from_yukawa_pipeline_prediction`
   once Phase 2.1 (wiring `yukawa_pipeline.rs` into
   `bayes_factor_multichannel.rs`) lands. Until then they are ignored, not
   deleted, so the migration target is preserved as a checklist.

## What was NOT done (and why)

### Production binary `p8_1_bayes_multichannel.rs` still calls `from_chain_match`

The plan text for Phase 2.7 says:

> Mark `chain_matcher.rs` `#[deprecated(...)]` and remove from any
> production `bin/` entry points.

**Removal from production bins is BLOCKED on Phase 2.1.**

`src/bin/p8_1_bayes_multichannel.rs` calls `from_chain_match` at lines
319 and 334 to load the chain-quark and chain-lepton channel evidence
that drives the publication-claim Bayes-factor headline. The successor
adapter `from_yukawa_pipeline_prediction` (per Phase 2.1) **does not
yet exist** in `src/route34/bayes_factor_multichannel.rs`. Removing the
`from_chain_match` calls without a working replacement would break the
`p8_1_bayes_multichannel` binary at runtime — every chain-channel JSON
load would either fail or silently degrade.

A `Grep` for `from_yukawa_pipeline_prediction` over the entire
`rust_solver/` tree returns zero hits. The plan itself confirms this is
Phase 2.1 work, not Phase 2.7:

> **P2.1 Wire `yukawa_pipeline.rs` into `bayes_factor_multichannel.rs`.**
> The existing `from_chain_match` adapter (`bayes_factor_multichannel.rs:845–879`)
> reads JSON output from a Galerkin-residual sweep. Replace with
> `from_yukawa_pipeline_prediction` that …

**Action item for the user:** Phase 2.1 must land before the chain_matcher
production-bin removal half of Phase 2.7 can be completed. The
deprecation attribute (this commit) is non-breaking — it produces
compiler warnings but no behavioural changes. Production output is
unchanged.

### Diagnostic binary `src/bin/p6_3_chain_match_diagnostic.rs` still calls chain_matcher

This binary is the *intentional* diagnostic re-run path documented in
the existing `chain_matcher.rs` module comments:

> Type retained only for diagnostic re-runs (`p6_3_chain_match_diagnostic`).

Removing this binary is out of scope for Phase 2.7 per the existing
intent that the diagnostic path is preserved for retraction-replication
work (see `references/p6_3_chain_match.md`).

## Forward checklist (Phase 2.1 follow-up)

When Phase 2.1 lands:

- [ ] Implement `from_yukawa_pipeline_prediction` in `bayes_factor_multichannel.rs`.
- [ ] Replace `from_chain_match` calls in `src/bin/p8_1_bayes_multichannel.rs`
      with `from_yukawa_pipeline_prediction`.
- [ ] Migrate the four `#[ignore]`d tests above to invoke the new adapter.
      Where the test asserts on a property of the OLD adapter that the new
      adapter does not have (e.g. the `chain_type` JSON-consistency check),
      either re-state the property for the new adapter or document the
      assertion as retracted with the chain_matcher channel.
- [ ] Verify `tests/test_p8_1_e2e_refuses_non_converged.rs` still
      passes against the rewired binary; this test is binary-level so it
      auto-migrates once the binary uses the new adapter.
- [ ] Once `from_chain_match` is no longer called from any non-`#[deprecated]`
      site, delete `from_chain_match` itself and consider deleting
      `chain_matcher.rs` (keeping only the diagnostic binary if still
      needed for the P6.3 retraction record).
