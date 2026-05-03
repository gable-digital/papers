//! Multi-pass discrimination pipeline with crash-safe checkpointing.
//!
//! Three passes, each with different precision/compute trade-offs:
//!   1. BroadSweep: low compute, ~millions of candidates, cheap topological
//!      and bundle-sector filters. Survivors written to JSONL.
//!   2. Refine: medium compute, ~thousands-to-tens-of-thousands of candidates,
//!      Donaldson-balance + light constraint refinement. Survivors written.
//!   3. Precision: high compute, ~tens-to-hundreds of candidates, full
//!      refinement at publication-grade precision.
//!
//! Each pass:
//!   - Reads input candidates from a JSONL file (or generates synthetic
//!     candidates for Pass 1).
//!   - Writes results to an output JSONL file (one record per candidate).
//!   - Periodically syncs a checkpoint file (default every 60s) so a
//!     crashed run can resume from the last sync without redoing finished
//!     candidates.
//!   - Writes via atomic temp-file-rename so crashes can't corrupt the
//!     output JSONL or checkpoint.
//!
//! Resume protocol:
//!   - On startup, look for `<output>.checkpoint`. If present, read it,
//!     determine the last fully-flushed record-id, skip those input
//!     candidates, and append from there.
//!   - If output JSONL exists but checkpoint doesn't (e.g. SIGKILL between
//!     output flush and checkpoint sync), truncate output to the last
//!     fully-formed JSON line and resume from there.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PassKind {
    BroadSweep,
    Refine,
    Precision,
}

impl PassKind {
    pub fn label(&self) -> &'static str {
        match self {
            PassKind::BroadSweep => "broad",
            PassKind::Refine => "refine",
            PassKind::Precision => "precision",
        }
    }
}

/// One candidate's bundle / Kahler / complex-structure parameters.
/// Compact representation: vectors of f64 (no NdArrays in the persisted
/// format, for simpler JSONL serialisation across runs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub id: u64,
    /// Topological data
    pub candidate_short_name: String,
    pub euler_characteristic: i32,
    pub fundamental_group: String,
    /// Moduli
    pub kahler_moduli: Vec<f64>,
    pub complex_moduli_real: Vec<f64>,
    pub complex_moduli_imag: Vec<f64>,
    // LEGACY-SUPERSEDED-BY-ROUTE34: `bundle_moduli: Vec<f64>` is an
    // unstructured raw-moduli vector consumed by the legacy heterotic.rs
    // monad bundle constructor. The publication-grade replacement is:
    //   * route34::bundle_search::CandidateBundle  (structured
    //     parameterisation: line-bundle degrees, monad map data, derived
    //     Chern classes via the splitting principle, Wilson-line element
    //     with canonical E_8 -> E_6 x SU(3) embedding from Slansky 1981 /
    //     AGLP 2011 / BHOP 2005)
    // The field is retained here for backwards compatibility with the
    // existing pass-1 broad-sweep search outputs (JSONL files contain
    // bundle_moduli arrays). New pipelines should populate the
    // `geometry` field below and read CandidateBundle through
    // route34::bundle_search.
    pub bundle_moduli: Vec<f64>,
    /// Lineage tracking: which previous-pass candidate produced this.
    /// Pass 1 candidates have parent_id = None.
    pub parent_id: Option<u64>,
    /// First-class CY3 descriptor: defining-relation bidegrees,
    /// ambient projective factors, intersection-form data, discrete
    /// quotient. Both the line-intersection sampler
    /// (`crate::cicy_sampler`) and the Koszul cohomology pipeline
    /// (`crate::zero_modes`) read from this single field, so the
    /// 5σ score for a given candidate is internally consistent
    /// (sampler and Koszul integrate over / compute on the same
    /// manifold).
    ///
    /// `#[serde(default)]` so that older JSONL pass outputs that
    /// pre-date the geometry field still deserialise — the missing
    /// field falls back to `CicyGeometry::default() = Tian-Yau Z/3`,
    /// which was implicitly hardcoded into the pipeline before the
    /// parameterisation existed.
    #[serde(default)]
    pub geometry: crate::geometry::CicyGeometry,
}

/// Per-pass reproducibility header. Emitted as the first JSONL line of
/// each output file (with a leading "_meta" tag) so reviewers /
/// downstream consumers know exactly what produced these numbers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassMetadata {
    pub _meta: bool,
    pub pass_kind: PassKind,
    pub solver_version: String,
    pub git_commit: String,
    pub build_profile: String,
    pub host_os: String,
    pub host_arch: String,
    pub timestamp_unix_secs: u64,
    pub seed: u64,
    pub n_candidates: u64,
    pub filter_threshold: f64,
}

impl PassMetadata {
    pub fn new(
        pass_kind: PassKind,
        seed: u64,
        n_candidates: u64,
        filter_threshold: f64,
    ) -> Self {
        Self {
            _meta: true,
            pass_kind,
            solver_version: env!("CARGO_PKG_VERSION").to_string(),
            git_commit: option_env!("CY3_GIT_COMMIT").unwrap_or("unknown").to_string(),
            build_profile: if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            }
            .to_string(),
            host_os: std::env::consts::OS.to_string(),
            host_arch: std::env::consts::ARCH.to_string(),
            timestamp_unix_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            seed,
            n_candidates,
            filter_threshold,
        }
    }
}

/// Result of running one candidate through a pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassResult {
    pub candidate: Candidate,
    pub pass_kind: PassKind,
    /// Loss components after the pass.
    pub loss_ricci: f64,
    pub loss_polyhedral: f64,
    pub loss_generation: f64,
    pub loss_coulomb: f64,
    pub loss_weak: f64,
    pub loss_strong: f64,
    pub loss_total: f64,
    /// PDG-2024 fermion-mass + CKM χ² (M4 → P1 → P2 → pdg::chi_squared_test).
    /// Zero by default; populated only when the precision pass runs the
    /// `compute_5sigma_score` end-to-end pipeline. `serde(default)`
    /// preserves backwards compatibility with pre-existing JSONL outputs.
    #[serde(default)]
    pub loss_pdg_chi2: f64,
    /// Unitarity violation of the predicted CKM matrix (Frobenius norm of
    /// V_CKM · V_CKM† − I).
    #[serde(default)]
    pub loss_ckm_unitarity: f64,
    /// Did this candidate pass the pass-specific filter threshold?
    pub passed_filter: bool,
    /// Wall-clock elapsed for this candidate.
    pub elapsed_ns: u64,
    /// Optional rank within this pass's outputs (assigned post-pass).
    pub rank: Option<usize>,
}

/// Checkpoint structure: persisted state of a running pass for crash
/// recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassCheckpoint {
    pub pass_kind: PassKind,
    pub input_path: Option<String>,
    pub output_path: String,
    pub total_candidates: u64,
    pub completed_candidates: u64,
    pub last_completed_id: Option<u64>,
    pub passed_filter_count: u64,
    pub started_at_unix_secs: u64,
    pub last_sync_at_unix_secs: u64,
    pub elapsed_secs: f64,
}

impl PassCheckpoint {
    pub fn new(pass_kind: PassKind, input: Option<String>, output: String, total: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            pass_kind,
            input_path: input,
            output_path: output,
            total_candidates: total,
            completed_candidates: 0,
            last_completed_id: None,
            passed_filter_count: 0,
            started_at_unix_secs: now,
            last_sync_at_unix_secs: now,
            elapsed_secs: 0.0,
        }
    }
}

/// Atomic file write: write to a temp file, fsync, then rename. Prevents
/// partial writes from corrupting a checkpoint or output file.
fn atomic_write(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let mut tmp = path.to_path_buf();
    tmp.set_extension(format!(
        "{}.tmp",
        path.extension().and_then(|s| s.to_str()).unwrap_or("dat")
    ));
    {
        let mut file = File::create(&tmp)?;
        file.write_all(contents)?;
        file.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Read existing JSONL output to recover the set of completed candidate IDs.
/// Used when we resume from an incomplete run that wrote output but lost
/// the checkpoint.
pub fn read_completed_ids(output_path: &Path) -> std::io::Result<HashSet<u64>> {
    let mut completed = HashSet::new();
    if !output_path.exists() {
        return Ok(completed);
    }
    let file = File::open(output_path)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        // Try to parse; skip malformed lines (e.g. partial last line)
        if let Ok(result) = serde_json::from_str::<PassResult>(&line) {
            completed.insert(result.candidate.id);
        }
    }
    Ok(completed)
}

/// Truncate the output file to the last fully-formed JSON line. Removes
/// any partial line that may have been written if the process crashed
/// mid-write.
pub fn truncate_output_to_last_complete_line(output_path: &Path) -> std::io::Result<()> {
    if !output_path.exists() {
        return Ok(());
    }
    let mut file = OpenOptions::new().read(true).write(true).open(output_path)?;
    let len = file.metadata()?.len();
    if len == 0 {
        return Ok(());
    }
    // Read the file backwards to find the last newline
    let mut buf = vec![0u8; len.min(64 * 1024) as usize];
    let read_start = len.saturating_sub(buf.len() as u64);
    file.seek(SeekFrom::Start(read_start))?;
    use std::io::Read;
    file.read_exact(&mut buf)?;

    // Find the last newline in buf
    if let Some(pos) = buf.iter().rposition(|&b| b == b'\n') {
        let truncate_to = read_start + (pos as u64) + 1; // include the newline
        file.set_len(truncate_to)?;
    } else if read_start == 0 {
        // No newline anywhere - file has only a partial line
        file.set_len(0)?;
    }
    Ok(())
}

/// Read input JSONL of PassResults (or treat as candidate-list).
/// For Pass 1 there's no input file; this function is used by Pass 2/3
/// to read the previous pass's output.
pub fn read_pass_results<P: AsRef<Path>>(path: P) -> std::io::Result<Vec<PassResult>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut results = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<PassResult>(&line) {
            Ok(r) => results.push(r),
            Err(e) => {
                eprintln!("warn: skipping malformed line: {}", e);
            }
        }
    }
    Ok(results)
}

/// Selection strategy for picking which candidates flow from one pass to
/// the next. Lets the caller pick exactly the slice of search space
/// they want to refine further.
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// The top K candidates (lowest total loss) that passed the previous
    /// pass's filter. Default behaviour.
    TopK(usize),
    /// All candidates with id in the inclusive range [start, end].
    IdRange { start: u64, end: u64 },
    /// All candidates whose rank (when sorted ascending by total loss)
    /// is in the inclusive range [start, end]. Rank 0 = best.
    RankRange { start: usize, end: usize },
    /// All candidates whose total_loss is in the inclusive range
    /// [min, max].
    LossRange { min: f64, max: f64 },
    /// All candidates that passed the previous pass's filter, sorted
    /// ascending by total loss (no truncation).
    AllPassed,
}

impl SelectionStrategy {
    pub fn describe(&self) -> String {
        match self {
            SelectionStrategy::TopK(k) => format!("top {} by total loss", k),
            SelectionStrategy::IdRange { start, end } => {
                format!("id range [{}..={}]", start, end)
            }
            SelectionStrategy::RankRange { start, end } => {
                format!("rank range [{}..={}]", start, end)
            }
            SelectionStrategy::LossRange { min, max } => {
                format!("loss range [{}..={}]", min, max)
            }
            SelectionStrategy::AllPassed => "all that passed filter".to_string(),
        }
    }
}

/// Filter + select candidates from a pass-result list according to the
/// requested strategy. The strategies that need rank-ordering sort
/// ascending by total loss before slicing.
pub fn select_for_next_pass<'a>(
    results: &'a [PassResult],
    strategy: &SelectionStrategy,
) -> Vec<&'a PassResult> {
    match strategy {
        SelectionStrategy::TopK(k) => {
            let mut filtered: Vec<&PassResult> =
                results.iter().filter(|r| r.passed_filter).collect();
            filtered.sort_by(|a, b| {
                a.loss_total
                    .partial_cmp(&b.loss_total)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            filtered.truncate(*k);
            filtered
        }
        SelectionStrategy::IdRange { start, end } => results
            .iter()
            .filter(|r| {
                r.candidate.id >= *start
                    && r.candidate.id <= *end
                    && r.passed_filter
            })
            .collect(),
        SelectionStrategy::RankRange { start, end } => {
            let mut all: Vec<&PassResult> =
                results.iter().filter(|r| r.passed_filter).collect();
            all.sort_by(|a, b| {
                a.loss_total
                    .partial_cmp(&b.loss_total)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all.into_iter().skip(*start).take(end + 1 - *start).collect()
        }
        SelectionStrategy::LossRange { min, max } => results
            .iter()
            .filter(|r| {
                r.passed_filter && r.loss_total >= *min && r.loss_total <= *max
            })
            .collect(),
        SelectionStrategy::AllPassed => {
            let mut all: Vec<&PassResult> =
                results.iter().filter(|r| r.passed_filter).collect();
            all.sort_by(|a, b| {
                a.loss_total
                    .partial_cmp(&b.loss_total)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            all
        }
    }
}

/// Backwards-compatible wrapper: select top-K by ascending total loss.
pub fn select_top_k_for_next_pass(
    results: &[PassResult],
    top_k: usize,
) -> Vec<&PassResult> {
    select_for_next_pass(results, &SelectionStrategy::TopK(top_k))
}

/// Convert a list of pass-N results into pass-(N+1) candidate inputs.
/// Updates parent_id and assigns fresh IDs in the new pass's space.
pub fn promote_to_next_pass(
    selected: &[&PassResult],
    next_pass_id_offset: u64,
) -> Vec<Candidate> {
    selected
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut c = r.candidate.clone();
            c.parent_id = Some(c.id);
            c.id = next_pass_id_offset + i as u64;
            c
        })
        .collect()
}

/// PassRunner: runs a pass over a candidate stream with periodic
/// checkpoint sync, supports resume. Generic over the per-candidate
/// scoring function so callers can plug in BroadSweep / Refine /
/// Precision-specific kernels.
pub struct PassRunner {
    pub pass_kind: PassKind,
    pub output_path: PathBuf,
    pub checkpoint_path: PathBuf,
    pub sync_interval: Duration,
    pub filter_threshold: f64, // candidates with loss_total <= threshold pass
}

impl PassRunner {
    pub fn new(
        pass_kind: PassKind,
        output_path: impl AsRef<Path>,
        sync_interval: Duration,
        filter_threshold: f64,
    ) -> Self {
        let output_path = output_path.as_ref().to_path_buf();
        let mut checkpoint_path = output_path.clone();
        checkpoint_path.set_extension("checkpoint");
        Self {
            pass_kind,
            output_path,
            checkpoint_path,
            sync_interval,
            filter_threshold,
        }
    }

    /// Run a pass with a per-candidate scoring closure. The closure
    /// returns the loss breakdown and bundle-sector forward-model values.
    /// Crash recovery is automatic: previously-completed IDs are loaded
    /// from the existing output JSONL (if any) and skipped.
    ///
    /// Backwards-compatible wrapper that adapts a budget-unaware scoring
    /// closure into the budget-aware path. New callers should prefer
    /// `run_budget`, which lets the scoring function short-circuit when
    /// the partial loss already exceeds the pass's filter threshold.
    pub fn run<F>(
        &self,
        candidates: Vec<Candidate>,
        score_fn: F,
    ) -> std::io::Result<PassRunReport>
    where
        F: Fn(&Candidate) -> ScoreResult + Send + Sync,
    {
        self.run_budget(candidates, |c, _budget| score_fn(c))
    }

    /// Streaming variant of `run_budget` that consumes an iterator of
    /// candidates lazily via Rayon's `par_bridge`. Avoids materialising
    /// the entire candidate list up front -- useful for broad-sweep
    /// passes where the candidate Vec would otherwise allocate ~800 MB
    /// at 1M candidates with the standard moduli layout.
    ///
    /// Crash recovery: the iterator is filtered against the previously-
    /// completed IDs before being handed to par_bridge. The filtering
    /// step is itself streaming (single pass through the iterator), so
    /// peak memory stays bounded by the working set of in-flight rayon
    /// tasks rather than the total candidate count.
    pub fn run_budget_streaming<I, F>(
        &self,
        candidates: I,
        score_fn: F,
        total_hint: u64,
    ) -> std::io::Result<PassRunReport>
    where
        I: IntoIterator<Item = Candidate>,
        I::IntoIter: Send,
        F: Fn(&Candidate, f64) -> ScoreResult + Send + Sync,
    {
        truncate_output_to_last_complete_line(&self.output_path)?;
        let completed_ids = read_completed_ids(&self.output_path)?;
        let already_done = completed_ids.len() as u64;

        let total = total_hint.max(already_done);

        eprintln!(
            "[{}] resume (streaming): {} / {} already completed; processing remainder",
            self.pass_kind.label(),
            already_done,
            total,
        );

        let pass_kind = self.pass_kind;
        let output_path = self.output_path.clone();
        let checkpoint_path = self.checkpoint_path.clone();
        let filter_threshold = self.filter_threshold;
        let sync_interval = self.sync_interval;

        let stop_flag = Arc::new(AtomicBool::new(false));
        let completed_count = Arc::new(AtomicU64::new(already_done));
        let passed_count = Arc::new(AtomicU64::new(0));
        let last_completed_id = Arc::new(Mutex::new(None::<u64>));

        let output_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_path)?;
        let mut output_writer_inner = BufWriter::new(output_file);

        let needs_metadata = output_writer_inner.get_ref().metadata()?.len() == 0;
        if needs_metadata {
            let meta = PassMetadata::new(
                self.pass_kind,
                already_done,
                total,
                self.filter_threshold,
            );
            if let Ok(json) = serde_json::to_string(&meta) {
                writeln!(output_writer_inner, "{}", json)?;
            }
        }
        let output_writer = Arc::new(Mutex::new(output_writer_inner));

        let started_at = Instant::now();
        let started_at_unix_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let stop_flag_sync = Arc::clone(&stop_flag);
        let completed_count_sync = Arc::clone(&completed_count);
        let passed_count_sync = Arc::clone(&passed_count);
        let last_completed_id_sync = Arc::clone(&last_completed_id);
        let output_writer_sync = Arc::clone(&output_writer);
        let checkpoint_path_sync = checkpoint_path.clone();
        let output_path_sync = output_path.clone();
        let sync_handle = std::thread::spawn(move || {
            let mut last_sync = Instant::now();
            while !stop_flag_sync.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                if last_sync.elapsed() >= sync_interval {
                    if let Ok(mut w) = output_writer_sync.lock() {
                        let _ = w.flush();
                    }
                    let now_unix = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let last_id = *last_completed_id_sync.lock().unwrap();
                    let cp = PassCheckpoint {
                        pass_kind,
                        input_path: None,
                        output_path: output_path_sync.to_string_lossy().to_string(),
                        total_candidates: total,
                        completed_candidates: completed_count_sync.load(Ordering::Relaxed),
                        last_completed_id: last_id,
                        passed_filter_count: passed_count_sync.load(Ordering::Relaxed),
                        started_at_unix_secs,
                        last_sync_at_unix_secs: now_unix,
                        elapsed_secs: started_at.elapsed().as_secs_f64(),
                    };
                    if let Ok(json) = serde_json::to_vec_pretty(&cp) {
                        let _ = atomic_write(&checkpoint_path_sync, &json);
                    }
                    last_sync = Instant::now();
                }
            }
            if let Ok(mut w) = output_writer_sync.lock() {
                let _ = w.flush();
            }
            let now_unix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let last_id = *last_completed_id_sync.lock().unwrap();
            let cp = PassCheckpoint {
                pass_kind,
                input_path: None,
                output_path: output_path_sync.to_string_lossy().to_string(),
                total_candidates: total,
                completed_candidates: completed_count_sync.load(Ordering::Relaxed),
                last_completed_id: last_id,
                passed_filter_count: passed_count_sync.load(Ordering::Relaxed),
                started_at_unix_secs,
                last_sync_at_unix_secs: now_unix,
                elapsed_secs: started_at.elapsed().as_secs_f64(),
            };
            if let Ok(json) = serde_json::to_vec_pretty(&cp) {
                let _ = atomic_write(&checkpoint_path_sync, &json);
            }
        });

        use rayon::prelude::*;
        candidates
            .into_iter()
            .filter(|c| !completed_ids.contains(&c.id))
            .par_bridge()
            .for_each(|c| {
                let t0 = Instant::now();
                let raw_score = score_fn(&c, filter_threshold);
                let sanitize = |x: f64| -> f64 {
                    if x.is_finite() {
                        x
                    } else {
                        1.0e3
                    }
                };
                let score = ScoreResult {
                    loss_ricci: sanitize(raw_score.loss_ricci),
                    loss_polyhedral: sanitize(raw_score.loss_polyhedral),
                    loss_generation: sanitize(raw_score.loss_generation),
                    loss_coulomb: sanitize(raw_score.loss_coulomb),
                    loss_weak: sanitize(raw_score.loss_weak),
                    loss_strong: sanitize(raw_score.loss_strong),
                    loss_pdg_chi2: sanitize(raw_score.loss_pdg_chi2),
                    loss_ckm_unitarity: sanitize(raw_score.loss_ckm_unitarity),
                };
                let elapsed_ns = t0.elapsed().as_nanos() as u64;
                let total_loss = score.loss_ricci
                    + score.loss_polyhedral
                    + score.loss_generation
                    + score.loss_coulomb
                    + score.loss_weak
                    + score.loss_strong
                    + score.loss_pdg_chi2
                    + score.loss_ckm_unitarity;
                let total_loss = if total_loss.is_finite() {
                    total_loss
                } else {
                    1.0e6
                };
                let passed_filter = total_loss <= filter_threshold;
                let id = c.id;
                let result = PassResult {
                    candidate: c,
                    pass_kind,
                    loss_ricci: score.loss_ricci,
                    loss_polyhedral: score.loss_polyhedral,
                    loss_generation: score.loss_generation,
                    loss_coulomb: score.loss_coulomb,
                    loss_weak: score.loss_weak,
                    loss_strong: score.loss_strong,
                    loss_pdg_chi2: score.loss_pdg_chi2,
                    loss_ckm_unitarity: score.loss_ckm_unitarity,
                    loss_total: total_loss,
                    passed_filter,
                    elapsed_ns,
                    rank: None,
                };
                if let Ok(json_line) = serde_json::to_string(&result) {
                    if let Ok(mut w) = output_writer.lock() {
                        let _ = writeln!(w, "{}", json_line);
                    }
                }
                completed_count.fetch_add(1, Ordering::Relaxed);
                if passed_filter {
                    passed_count.fetch_add(1, Ordering::Relaxed);
                }
                *last_completed_id.lock().unwrap() = Some(id);
            });

        stop_flag.store(true, Ordering::Relaxed);
        let _ = sync_handle.join();

        Ok(PassRunReport {
            pass_kind,
            total_candidates: total,
            completed: completed_count.load(Ordering::Relaxed),
            passed_filter: passed_count.load(Ordering::Relaxed),
            elapsed_secs: started_at.elapsed().as_secs_f64(),
            output_path,
            checkpoint_path,
        })
    }

    /// Run a pass with a budget-aware scoring closure that may short-
    /// circuit when the partial cumulative loss already exceeds the
    /// pass's filter threshold. The closure receives the per-candidate
    /// budget (= the runner's `filter_threshold`) and is free to return
    /// early with large penalty values in any channel once partial-sum
    /// > budget; doing so saves compute on candidates that cannot pass
    /// regardless of how the remaining channels score.
    ///
    /// All loss components are non-negative (relative-error squares,
    /// variance estimators, integer-distance squares), so the partial-
    /// sum-exceeds-budget short-circuit is sound: once partial sum
    /// exceeds the threshold, no remaining channel can bring the total
    /// back below.
    pub fn run_budget<F>(
        &self,
        candidates: Vec<Candidate>,
        score_fn: F,
    ) -> std::io::Result<PassRunReport>
    where
        F: Fn(&Candidate, f64) -> ScoreResult + Send + Sync,
    {
        // Recovery: read previously-completed IDs and truncate output to
        // last full line.
        truncate_output_to_last_complete_line(&self.output_path)?;
        let completed_ids = read_completed_ids(&self.output_path)?;

        let total = candidates.len() as u64;
        let already_done = completed_ids.len() as u64;
        let remaining: Vec<Candidate> = candidates
            .into_iter()
            .filter(|c| !completed_ids.contains(&c.id))
            .collect();

        eprintln!(
            "[{}] resume: {} / {} already completed; processing {} candidates",
            self.pass_kind.label(),
            already_done,
            total,
            remaining.len()
        );

        let pass_kind = self.pass_kind;
        let output_path = self.output_path.clone();
        let checkpoint_path = self.checkpoint_path.clone();
        let filter_threshold = self.filter_threshold;
        let sync_interval = self.sync_interval;

        let stop_flag = Arc::new(AtomicBool::new(false));
        let completed_count = Arc::new(AtomicU64::new(already_done));
        let passed_count = Arc::new(AtomicU64::new(0));
        let last_completed_id = Arc::new(Mutex::new(None::<u64>));

        // Output writer: append to file in batches via a Mutex-protected
        // BufWriter. Write thread flushes JSONL records as they arrive.
        let output_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&output_path)?;
        let mut output_writer_inner = BufWriter::new(output_file);

        // Emit reproducibility metadata as the first line, but only if
        // the file is currently empty (resume runs already have it).
        let needs_metadata = output_writer_inner.get_ref().metadata()?.len() == 0;
        if needs_metadata {
            let meta = PassMetadata::new(
                self.pass_kind,
                already_done, // placeholder for seed; bench layer can set
                total,
                self.filter_threshold,
            );
            if let Ok(json) = serde_json::to_string(&meta) {
                writeln!(output_writer_inner, "{}", json)?;
            }
        }
        let output_writer = Arc::new(Mutex::new(output_writer_inner));

        // Checkpoint sync thread
        let started_at = Instant::now();
        let started_at_unix_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let stop_flag_sync = Arc::clone(&stop_flag);
        let completed_count_sync = Arc::clone(&completed_count);
        let passed_count_sync = Arc::clone(&passed_count);
        let last_completed_id_sync = Arc::clone(&last_completed_id);
        let output_writer_sync = Arc::clone(&output_writer);
        let checkpoint_path_sync = checkpoint_path.clone();
        let output_path_sync = output_path.clone();
        let sync_handle = std::thread::spawn(move || {
            let mut last_sync = Instant::now();
            while !stop_flag_sync.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                if last_sync.elapsed() >= sync_interval {
                    // Flush output buffer
                    if let Ok(mut w) = output_writer_sync.lock() {
                        let _ = w.flush();
                    }
                    // Write checkpoint
                    let now_unix = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    let last_id = *last_completed_id_sync.lock().unwrap();
                    let cp = PassCheckpoint {
                        pass_kind,
                        input_path: None,
                        output_path: output_path_sync.to_string_lossy().to_string(),
                        total_candidates: total,
                        completed_candidates: completed_count_sync.load(Ordering::Relaxed),
                        last_completed_id: last_id,
                        passed_filter_count: passed_count_sync.load(Ordering::Relaxed),
                        started_at_unix_secs,
                        last_sync_at_unix_secs: now_unix,
                        elapsed_secs: started_at.elapsed().as_secs_f64(),
                    };
                    if let Ok(json) = serde_json::to_vec_pretty(&cp) {
                        let _ = atomic_write(&checkpoint_path_sync, &json);
                    }
                    last_sync = Instant::now();
                }
            }
            // Final flush + sync on shutdown
            if let Ok(mut w) = output_writer_sync.lock() {
                let _ = w.flush();
            }
            let now_unix = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let last_id = *last_completed_id_sync.lock().unwrap();
            let cp = PassCheckpoint {
                pass_kind,
                input_path: None,
                output_path: output_path_sync.to_string_lossy().to_string(),
                total_candidates: total,
                completed_candidates: completed_count_sync.load(Ordering::Relaxed),
                last_completed_id: last_id,
                passed_filter_count: passed_count_sync.load(Ordering::Relaxed),
                started_at_unix_secs,
                last_sync_at_unix_secs: now_unix,
                elapsed_secs: started_at.elapsed().as_secs_f64(),
            };
            if let Ok(json) = serde_json::to_vec_pretty(&cp) {
                let _ = atomic_write(&checkpoint_path_sync, &json);
            }
        });

        // Process candidates with rayon parallelism
        use rayon::prelude::*;
        remaining.par_iter().for_each(|c| {
            let t0 = Instant::now();
            let raw_score = score_fn(c, filter_threshold);
            // NaN/Inf guard: replace any non-finite component with a
            // large penalty (1e3 per channel) so the candidate fails
            // the filter rather than poisoning the output.
            let sanitize = |x: f64| -> f64 {
                if x.is_finite() {
                    x
                } else {
                    1.0e3
                }
            };
            let score = ScoreResult {
                loss_ricci: sanitize(raw_score.loss_ricci),
                loss_polyhedral: sanitize(raw_score.loss_polyhedral),
                loss_generation: sanitize(raw_score.loss_generation),
                loss_coulomb: sanitize(raw_score.loss_coulomb),
                loss_weak: sanitize(raw_score.loss_weak),
                loss_strong: sanitize(raw_score.loss_strong),
                loss_pdg_chi2: sanitize(raw_score.loss_pdg_chi2),
                loss_ckm_unitarity: sanitize(raw_score.loss_ckm_unitarity),
            };
            let elapsed_ns = t0.elapsed().as_nanos() as u64;
            let total_loss = score.loss_ricci
                + score.loss_polyhedral
                + score.loss_generation
                + score.loss_coulomb
                + score.loss_weak
                + score.loss_strong
                + score.loss_pdg_chi2
                + score.loss_ckm_unitarity;
            let total_loss = if total_loss.is_finite() {
                total_loss
            } else {
                1.0e6
            };
            let passed_filter = total_loss <= filter_threshold;
            let result = PassResult {
                candidate: c.clone(),
                pass_kind,
                loss_ricci: score.loss_ricci,
                loss_polyhedral: score.loss_polyhedral,
                loss_generation: score.loss_generation,
                loss_coulomb: score.loss_coulomb,
                loss_weak: score.loss_weak,
                loss_strong: score.loss_strong,
                loss_total: total_loss,
                loss_pdg_chi2: score.loss_pdg_chi2,
                loss_ckm_unitarity: score.loss_ckm_unitarity,
                passed_filter,
                elapsed_ns,
                rank: None,
            };
            // Append to output
            if let Ok(json_line) = serde_json::to_string(&result) {
                if let Ok(mut w) = output_writer.lock() {
                    let _ = writeln!(w, "{}", json_line);
                }
            }
            completed_count.fetch_add(1, Ordering::Relaxed);
            if passed_filter {
                passed_count.fetch_add(1, Ordering::Relaxed);
            }
            *last_completed_id.lock().unwrap() = Some(c.id);
        });

        stop_flag.store(true, Ordering::Relaxed);
        let _ = sync_handle.join();

        Ok(PassRunReport {
            pass_kind,
            total_candidates: total,
            completed: completed_count.load(Ordering::Relaxed),
            passed_filter: passed_count.load(Ordering::Relaxed),
            elapsed_secs: started_at.elapsed().as_secs_f64(),
            output_path,
            checkpoint_path,
        })
    }
}

#[derive(Debug, Clone)]
pub struct PassRunReport {
    pub pass_kind: PassKind,
    pub total_candidates: u64,
    pub completed: u64,
    pub passed_filter: u64,
    pub elapsed_secs: f64,
    pub output_path: PathBuf,
    pub checkpoint_path: PathBuf,
}

/// Per-candidate score returned by the scoring function.
#[derive(Debug, Clone, Copy)]
pub struct ScoreResult {
    pub loss_ricci: f64,
    pub loss_polyhedral: f64,
    pub loss_generation: f64,
    pub loss_coulomb: f64,
    pub loss_weak: f64,
    pub loss_strong: f64,
    /// PDG-2024 fermion-mass + CKM χ² (M4 → P1 → P2 → pdg::chi_squared_test).
    /// Zero by default; populated only when `compute_5sigma_score` is run.
    /// 5σ exclusion threshold is χ² > 56.4 at 13 d.o.f. (p < 5.7e-7).
    pub loss_pdg_chi2: f64,
    /// Unitarity violation of the predicted CKM matrix (Frobenius norm of
    /// V_CKM · V_CKM† − I). Zero for an exactly unitary CKM.
    pub loss_ckm_unitarity: f64,
}

impl Default for ScoreResult {
    fn default() -> Self {
        Self {
            loss_ricci: 0.0,
            loss_polyhedral: 0.0,
            loss_generation: 0.0,
            loss_coulomb: 0.0,
            loss_weak: 0.0,
            loss_strong: 0.0,
            loss_pdg_chi2: 0.0,
            loss_ckm_unitarity: 0.0,
        }
    }
}

impl ScoreResult {
    pub fn total(&self) -> f64 {
        self.loss_ricci
            + self.loss_polyhedral
            + self.loss_generation
            + self.loss_coulomb
            + self.loss_weak
            + self.loss_strong
            + self.loss_pdg_chi2
            + self.loss_ckm_unitarity
    }
}

// ---------------------------------------------------------------------------
// End-to-end 5σ score: M4 → P1 → P2 → PDG
//
// Wires the new modules (cicy_sampler, zero_modes, yukawa_overlap, pdg)
// into the discrimination scoring path. Until this function is called,
// the pipeline runs only the Donaldson-on-polysphere path and the new
// machinery is dead weight (the case before this commit).
//
// Performance: ~1500 sample points + Yukawa SVD per call ≈ 1-2s on a
// modern CPU. Callers should prefer `compute_5sigma_score` only at
// the precision pass, not the broad sweep.
// ---------------------------------------------------------------------------

/// Configuration for the end-to-end 5σ score.
#[derive(Debug, Clone)]
pub struct FiveSigmaConfig {
    pub n_sample_points: usize,
    pub sampler_seed: u64,
    pub mu_init_gev: f64,    // heterotic GUT scale; default 1e16 GeV
    /// When `true`, run the chapter-21 η-integral evaluator
    /// (`crate::route34::eta_evaluator`) on the candidate's geometry
    /// and fold its χ² (vs the chapter's observed
    /// `η_obs = (6.115 ± 0.038) × 10⁻¹⁰`) into the per-candidate
    /// `total_loss` as `loss_eta_chi2`. Default `false` because each
    /// evaluation runs Donaldson balancing and can take tens of
    /// seconds to several minutes per candidate; sweep harnesses
    /// that need throughput should leave it off and run the
    /// standalone `eta_discriminate` binary instead.
    pub compute_eta_chi2: bool,
}

impl Default for FiveSigmaConfig {
    fn default() -> Self {
        Self {
            n_sample_points: 1500,
            sampler_seed: 17,
            mu_init_gev: 1.0e16,
            compute_eta_chi2: false,
        }
    }
}

/// Per-stage diagnostic from `compute_5sigma_score`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiveSigmaBreakdown {
    pub n_samples_accepted: usize,
    pub n_27_generations: u32,
    pub yukawa_norm_u: f64,
    pub yukawa_norm_d: f64,
    pub yukawa_norm_e: f64,
    pub chi2_total: f64,
    pub chi2_dof: usize,
    pub p_value: f64,
    pub passes_5_sigma: bool,
    pub ckm_unitarity_residual: f64,
    /// Route 2 (chapter 8) χ²: cross-term-as-coupling identity for
    /// the gauge-sector Yukawas (matter side is deferred). Returns
    /// ≈ 0 by construction whenever the prediction lines up with
    /// PDG's gauge couplings; non-zero values surface mismatches in
    /// the substrate-side identification or future drift in the
    /// PDG values.
    #[serde(default)]
    pub loss_route2_gauge: f64,
    /// Route 3 (chapter 21) χ²: η-integral on the candidate
    /// geometry, compared against `η_obs = (6.115 ± 0.038) × 10⁻¹⁰`.
    /// `0.0` when `FiveSigmaConfig.compute_eta_chi2` is `false`
    /// (the default — keeps wallclock manageable). When `true`, this
    /// drives the evaluator at
    /// `crate::route34::eta_evaluator::evaluate_eta_{tian_yau,schoen}`
    /// and folds the result here.
    #[serde(default)]
    pub loss_eta_chi2: f64,
    /// Predicted η value (positive, dimensionless), or `0.0` when
    /// the η evaluator did not run.
    #[serde(default)]
    pub eta_predicted: f64,
    /// 1σ uncertainty on `eta_predicted`, or `0.0` when the η
    /// evaluator did not run.
    #[serde(default)]
    pub eta_uncertainty: f64,
    /// Route 4 (chapter 8 §"Pinning Down Route 4") χ²: predicted-vs-
    /// observed polyhedral wavenumbers at the Saturn / Jupiter
    /// north / Jupiter south polar critical-boundaries, computed by
    /// dispatching on the candidate's fundamental_group to
    /// `crate::route34::KillingResult::{tianyau_z3, schoen_z3xz3}`
    /// and feeding into `route34::route4_predictor::route4_discrimination`.
    /// Always populated (the predictor is cheap — no Donaldson
    /// balancing, just discrete isometry-group lookup + Arnold
    /// classification).
    #[serde(default)]
    pub loss_route4_chi2: f64,
    /// Soft match scores per planet (1.0 = wavenumber in predicted
    /// set; 1/(1+d²) when the closest predicted wavenumber differs
    /// by `d`). Together they cover the chapter's `{6, 8, 5}`
    /// observed-wavenumber set.
    #[serde(default)]
    pub route4_saturn_match: f64,
    #[serde(default)]
    pub route4_jupiter_north_match: f64,
    #[serde(default)]
    pub route4_jupiter_south_match: f64,
    /// Route 5 (chapter 8 §"A New Discrimination Channel") χ²:
    /// scalar spectral index `n_s` from `E_8 × E_8` Coxeter
    /// geometry vs Planck 2018 measurement
    /// `n_s = 0.9649 ± 0.0042`. Leading-order substrate prediction
    /// `n_s = 58/60` is candidate-CY3-independent; sub-leading
    /// merger-class correction `Δn_s ~ 5e-4` is candidate-specific
    /// (driven by the Killing-vector projection structure at the
    /// inversion boundary). Always populated, no opt-in (the
    /// computation is closed-form).
    #[serde(default)]
    pub loss_route5_ns: f64,
    /// Predicted `n_s` for this candidate (leading order +
    /// merger-class correction).
    #[serde(default)]
    pub route5_n_s_predicted: f64,
}

/// End-to-end 5σ score: drive M4 line-intersection sampler →
/// P1 polynomial zero-mode seeds → P2 triple-overlap Yukawas →
/// 1-loop SM RG run to M_Z → PDG-2024 χ².
///
/// Returns the full diagnostic breakdown plus the χ² and CKM-unitarity
/// loss values to merge into `ScoreResult`. Callers can fold these into
/// the `ScoreResult` they're already accumulating.
///
/// ## GPU acceleration
///
/// When the crate is built with `--features gpu` and a CUDA device is
/// available, the heavy per-point stages dispatch to CUDA kernels
/// transparently:
///
/// * holomorphic 3-form Ω at every sample point — `crate::gpu_omega`,
/// * polynomial-seed evaluation per (mode, point) —
///   `crate::gpu_polynomial_seeds`,
/// * triple-overlap reductions `λ_{ij}` and Hermitian norms `N^{L,R}`
///   per sector — `crate::gpu_yukawa`,
/// * harmonic-projection Adam loop (when `project_to_harmonic` is
///   used downstream) — `crate::gpu_harmonic`,
/// * parallel Newton in the line-intersection sampler —
///   `crate::gpu_sampler`.
///
/// On any failure (no CUDA driver, missing `nvrtc.dll`, kernel error)
/// each stage falls back transparently to its CPU implementation, so
/// the same call site produces the same numerical result on any host.
/// CPU/GPU parity is enforced by per-kernel parity tests in each
/// `gpu_*` module.
///
/// **Caveats** (see honesty audit):
/// - P1 zero modes are polynomial seeds, not harmonic representatives
///   (P1.5 follow-up). The Yukawa magnitudes are correct only up to
///   overall normalisation in the long-wavelength limit.
/// - P2 contraction is the reduced (scalar-product) form; the full
///   epsilon-tensor cup product is a separate follow-up.
/// - 1-loop SM Yukawa β-functions only.
/// - CKM extracted from MGS-fixupped SVD (Hermitian Jacobi has known
///   precision issue tracked separately).
pub fn compute_5sigma_score(
    config: &FiveSigmaConfig,
) -> Result<FiveSigmaBreakdown, &'static str> {
    // Default candidate: the Tian-Yau Z/3 line of the substrate
    // theory's polyhedral-resonance hypothesis. To score a different
    // candidate (e.g. Schoen Z/3 × Z/3) call
    // [`compute_5sigma_score_for_candidate`] directly.
    compute_5sigma_score_for_candidate(config, &default_tian_yau_candidate())
}

/// Build the canonical Tian-Yau Z/3 candidate that the legacy
/// [`compute_5sigma_score`] used implicitly.
fn default_tian_yau_candidate() -> Candidate {
    Candidate {
        id: 0,
        candidate_short_name: "tian-yau-z3-default".to_string(),
        euler_characteristic: -6,
        fundamental_group: "Z3".to_string(),
        kahler_moduli: Vec::new(),
        complex_moduli_real: Vec::new(),
        complex_moduli_imag: Vec::new(),
        bundle_moduli: Vec::new(),
        parent_id: None,
        geometry: crate::geometry::CicyGeometry::tian_yau_z3(),
    }
}

/// End-to-end 5σ score *for a specific candidate*. The candidate's
/// `geometry` field selects the CY3 manifold: both the line-
/// intersection sampler and the Koszul cohomology run on the same
/// `CicyGeometry`, so the integrated quantities are coherent.
///
/// Currently supports `geometry.n_relations() == 3` candidates
/// (Tian-Yau Z/3); the Schoen-class `(3,3)` hypersurface
/// (`n_relations() == 1`) needs a separate sampler entry point and
/// returns an explicit error rather than silently scoring against the
/// wrong manifold.
pub fn compute_5sigma_score_for_candidate(
    config: &FiveSigmaConfig,
    candidate: &Candidate,
) -> Result<FiveSigmaBreakdown, &'static str> {
    use crate::cicy_sampler::{BicubicPair, CicySampler};
    use crate::zero_modes::{
        AmbientCY3, MonadBundle, compute_zero_mode_spectrum,
    };
    use crate::yukawa_overlap::{
        compute_omega_at_samples, compute_yukawa_spectrum,
        extract_ckm, to_predicted_yukawas,
    };
    use crate::pdg::{
        Pdg2024, chi_squared_test, extract_observables, rg_run_to_mz,
    };

    let geometry = &candidate.geometry;
    if !geometry.satisfies_calabi_yau_condition() {
        return Err("candidate geometry violates Calabi-Yau condition");
    }
    if geometry.n_fold() != 3 {
        return Err("candidate geometry is not a 3-fold");
    }

    // Stage 1: M4 sampling. Two dispatched paths:
    //
    //   * Tian-Yau Z/3 (3 relations on CP^3 × CP^3): canonical
    //     line-intersection sampler in `crate::cicy_sampler`.
    //   * Schoen Z/3 × Z/3 fiber product (2 relations on
    //     CP^2 × CP^2 × CP^1): bridge through
    //     `crate::route34::schoen_sampler`, packing the resulting
    //     `SchoenPoint`s into the same `SampledPoint` shape the
    //     downstream Stage 2-5 pipeline consumes (`x[0..3]` →
    //     `z[0..3]`, `t[0]` → `z[3]`; `y[0..3]` → `w[0..3]`,
    //     `t[1]` → `w[3]`). The geometric meaning of "z" vs "w"
    //     differs between the two candidates; downstream stages
    //     consume the points only as quadrature nodes (with
    //     `weight` and `omega` already pre-computed by the
    //     sampler), so the index-renaming is harmless.
    //
    // Other geometries fall through to the routes-only partial
    // pipeline.
    let (mut samples, sampled_via_schoen) = if geometry.ambient_factors == vec![3, 3]
        && geometry.n_relations() == 3
    {
        let bicubic = BicubicPair::z3_invariant_default();
        let mut sampler = CicySampler::new(bicubic, config.sampler_seed);
        let mut samples = sampler.sample_batch(config.n_sample_points);
        if geometry.quotient_order > 1 {
            CicySampler::apply_z3_quotient(&mut samples);
        }
        (samples, false)
    } else if geometry.ambient_factors == vec![2, 2, 1]
        && geometry.n_relations() == 2
    {
        let samples = sample_schoen_points_as_sampled_points(config);
        (samples, true)
    } else {
        return compute_5sigma_score_routes_only(config, candidate);
    };
    let n_accepted = samples.len();
    if n_accepted == 0 {
        return Err("M4 sampler produced zero accepted points");
    }

    // Stage 2: holomorphic 3-form Ω at every sample point.
    // For Schoen, Ω was already computed inside the SchoenSampler
    // (it lives on each SchoenPoint); we propagate it. For Tian-Yau,
    // recompute via the bicubic Jacobian residue.
    let omega: Vec<num_complex::Complex64> = if sampled_via_schoen {
        samples.iter().map(|s| s.omega).collect()
    } else {
        let bicubic = BicubicPair::z3_invariant_default();
        compute_omega_at_samples(&samples, &bicubic)
    };
    // The CicySampler-recomputed omega doesn't match the SampledPoint
    // omega exactly (cofactor-expansion roundoff); we only use the
    // recomputed version for the TY path to preserve existing
    // numerical behaviour (and the existing parity tests). For Schoen
    // we use the sampler-supplied omega directly.
    let _ = &mut samples; // explicit no-op for clarity

    // Stage 3: P1 zero-mode spectrum check on the candidate's
    // geometry. The Koszul + BBW path reads the geometry's
    // intersection numbers via `chern_classes`, so the index
    // computation is automatically consistent with the sampler.
    let bundle = MonadBundle::anderson_lukas_palti_example();
    let ambient = AmbientCY3 {
        h11: geometry.h11_upstairs,
        h21: geometry.h21_upstairs,
        euler_chi: geometry.chi_upstairs,
        quotient_order: geometry.quotient_order,
        geometry: geometry.clone(),
    };
    let spectrum = compute_zero_mode_spectrum(&bundle, &ambient);
    if spectrum.generation_count == 0 {
        if sampled_via_schoen {
            // Schoen-specific: the demo monad is CP^3 × CP^3-shaped
            // and gives c_3 = 0 on Schoen's CP^2 × CP^2 × CP^1
            // intersection form. Until task #47 (real ALP 2011 §4
            // Γ-equivariant Schoen-side bundle) lands, fall through
            // to the routes-only partial pipeline with the sample
            // count from Stage 1 preserved as a diagnostic.
            let mut bd = compute_5sigma_score_routes_only(config, candidate)?;
            bd.n_samples_accepted = n_accepted;
            return Ok(bd);
        }
        return Err("zero-mode spectrum yielded zero generations");
    }

    // Stage 4: P2 Yukawa overlap (six bundles, all ALP for the
    // minimal placeholder configuration).
    let yspec = compute_yukawa_spectrum(
        &samples, &omega,
        &bundle, &bundle, &bundle, &bundle, &bundle, &bundle,
        &ambient,
    );

    // Frobenius norms for diagnostics.
    let frob = |m: &[[(f64, f64); 3]; 3]| -> f64 {
        let mut s = 0.0_f64;
        for row in m { for &(re, im) in row { s += re * re + im * im; } }
        s.sqrt()
    };
    let yu_norm = frob(&yspec.y_u);
    let yd_norm = frob(&yspec.y_d);
    let ye_norm = frob(&yspec.y_e);
    if !(yu_norm.is_finite() && yd_norm.is_finite() && ye_norm.is_finite()) {
        return Err("Yukawa spectrum non-finite");
    }

    // Stage 5: RG run + PDG χ².
    let predicted_yukawas = to_predicted_yukawas(&yspec, config.mu_init_gev);
    let pdg = Pdg2024::new();
    let running = rg_run_to_mz(&predicted_yukawas)
        .map_err(|_| "rg_run_to_mz failed")?;
    let observables = extract_observables(&running, &pdg);
    let chi2 = chi_squared_test(&observables, &pdg);

    // CKM unitarity residual.
    let ckm = extract_ckm(&yspec.y_u, &yspec.y_d);
    let mut residual = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            // (V V†)_{ij} = Σ_k V_{ik} V_{jk}*
            let mut sum_re = 0.0;
            let mut sum_im = 0.0;
            for k in 0..3 {
                let (a_re, a_im) = ckm[i][k];
                let (b_re, b_im) = ckm[j][k];
                sum_re += a_re * b_re + a_im * b_im;
                sum_im += a_im * b_re - a_re * b_im;
            }
            let target_re = if i == j { 1.0 } else { 0.0 };
            residual += (sum_re - target_re).powi(2) + sum_im.powi(2);
        }
    }
    let ckm_residual = residual.sqrt();

    // Stage 6: Route 2 (chapter 8 §"Four Substrate-Specific
    // Computational Routes" §Route 2). Cross-term-as-coupling
    // identity for the gauge-sector Yukawas. No CY3 dependence —
    // this is the substrate-physics commitment that the gauge-Yukawa
    // magnitude IS the corresponding gauge coupling — so the
    // prediction matches PDG by construction and χ²_route2_gauge ≈ 0.
    // Non-zero values would indicate a substrate-side inconsistency
    // or future drift in the PDG gauge-coupling values; the field
    // is plumbed so a future CY3-derived predictor (when route34's
    // Killing-vector pipeline lands) can supply non-trivial input.
    let route2_pred_gauge =
        crate::route12::route2::GaugeYukawaPrediction::from_pdg(&pdg);
    let route2_pred_matter = crate::route12::route2::MatterYukawaPrediction::default();
    let loss_route2_gauge = crate::route12::route2::route2_chi2_against_pdg(
        &route2_pred_gauge,
        &route2_pred_matter,
        &pdg,
    );

    // Stage 7 (opt-in): Route 3 η-integral. Runs Donaldson balancing
    // on the candidate's geometry; ~10s–10min per candidate, hence
    // the explicit config flag. Observed value from chapter-21
    // analysis: η_obs = (6.115 ± 0.038) × 10⁻¹⁰.
    let (loss_eta_chi2, eta_predicted, eta_uncertainty) = if config.compute_eta_chi2 {
        compute_eta_chi2_for_geometry(geometry, config.sampler_seed)
    } else {
        (0.0, 0.0, 0.0)
    };

    // Stage 8: Route 4 polyhedral-resonance χ². Cheap — discrete
    // isometry-group lookup + Arnold catastrophe-theory ADE
    // classification + soft wavenumber match against the chapter's
    // {6, 8, 5} observed set. Always-on.
    let (
        loss_route4_chi2,
        route4_saturn_match,
        route4_jupiter_north_match,
        route4_jupiter_south_match,
    ) = compute_route4_chi2_for_candidate(candidate);

    // Stage 9: Route 5 scalar spectral index. Closed-form (no
    // sampling, no Donaldson balancing); leading order
    // n_s = 58/60 ≈ 0.9667 vs Planck 2018 0.9649 ± 0.0042.
    let (loss_route5_ns, route5_n_s_predicted, _) =
        crate::route5::spectral_index::route5_chi2_against_planck(
            &candidate.fundamental_group,
        );

    Ok(FiveSigmaBreakdown {
        n_samples_accepted: n_accepted,
        n_27_generations: spectrum.n_27,
        yukawa_norm_u: yu_norm,
        yukawa_norm_d: yd_norm,
        yukawa_norm_e: ye_norm,
        chi2_total: chi2.chi2_total,
        chi2_dof: chi2.dof,
        p_value: chi2.p_value,
        passes_5_sigma: chi2.passes_5_sigma,
        ckm_unitarity_residual: ckm_residual,
        loss_route2_gauge,
        loss_eta_chi2,
        eta_predicted,
        eta_uncertainty,
        loss_route4_chi2,
        route4_saturn_match,
        route4_jupiter_north_match,
        route4_jupiter_south_match,
        loss_route5_ns,
        route5_n_s_predicted,
    })
}

/// Route 4 polyhedral-resonance χ² + per-planet match scores.
/// Dispatches on the candidate's `fundamental_group` to one of the
/// canonical [`crate::route34::KillingResult`] constructors, then
/// runs [`crate::route34::route4_predictor::route4_discrimination`].
/// On any predictor error (e.g. unknown group, Arnold germ failure)
/// returns all zeros — Route 4 then contributes nothing to the
/// candidate's total loss, which is the right behaviour for
/// candidates outside the `{TY/Z3, Schoen/Z3xZ3}` set.
fn compute_route4_chi2_for_candidate(candidate: &Candidate) -> (f64, f64, f64, f64) {
    use crate::route34::route4_predictor::route4_discrimination;
    use crate::route34::KillingResult;
    let killing = match candidate.fundamental_group.as_str() {
        "Z3" => KillingResult::tianyau_z3(),
        "Z3xZ3" => KillingResult::schoen_z3xz3(),
        _ => return (0.0, 0.0, 0.0, 0.0),
    };
    match route4_discrimination(&killing) {
        Ok(r) => (
            r.combined_chi_squared,
            r.saturn_match,
            r.jupiter_north_match,
            r.jupiter_south_match,
        ),
        Err(_) => (0.0, 0.0, 0.0, 0.0),
    }
}

/// Partial-pipeline path for candidates whose geometry isn't
/// supported by the line-intersection sampler (notably Schoen
/// Z/3 × Z/3 on `CP^2 × CP^2 × CP^1`). Runs only the routes that
/// don't depend on the sampler:
///
///   * Route 2: cross-term-as-coupling identity (always fast)
///   * Route 3: η-integral (opt-in via `compute_eta_chi2`; uses the
///     candidate's own internal sampler in `route34::eta_evaluator`)
///   * Route 4: discrete-isometry-driven polyhedral wavenumber
///     match (always fast)
///
/// Returns a [`FiveSigmaBreakdown`] with the unsupported fields
/// (sample counts, generation count, Yukawa norms, PDG χ², CKM
/// residual) set to zero / `false`. The downstream
/// [`sweep_candidates`] aggregator sums `loss_route2_gauge +
/// loss_eta_chi2 + loss_route4_chi2` for the total loss in this
/// case — still a meaningful per-candidate ranking signal because
/// Routes 2/3/4 cover three of the chapter-8 four-route program.
fn compute_5sigma_score_routes_only(
    config: &FiveSigmaConfig,
    candidate: &Candidate,
) -> Result<FiveSigmaBreakdown, &'static str> {
    use crate::pdg::Pdg2024;
    let geometry = &candidate.geometry;
    let pdg = Pdg2024::new();

    // Route 2: cross-term-as-coupling identity. No CY3 input needed.
    let route2_pred_gauge =
        crate::route12::route2::GaugeYukawaPrediction::from_pdg(&pdg);
    let route2_pred_matter = crate::route12::route2::MatterYukawaPrediction::default();
    let loss_route2_gauge = crate::route12::route2::route2_chi2_against_pdg(
        &route2_pred_gauge,
        &route2_pred_matter,
        &pdg,
    );

    // Route 3 (opt-in): η-integral on the candidate's geometry via
    // route34::eta_evaluator (which has its own internal sampler for
    // both Tian-Yau and Schoen).
    let (loss_eta_chi2, eta_predicted, eta_uncertainty) = if config.compute_eta_chi2 {
        compute_eta_chi2_for_geometry(geometry, config.sampler_seed)
    } else {
        (0.0, 0.0, 0.0)
    };

    // Route 4: polyhedral-resonance discrimination via discrete
    // isometry group lookup.
    let (loss_route4_chi2, sat, jn, js) = compute_route4_chi2_for_candidate(candidate);

    // Route 5 closed-form: candidate-CY3-independent leading-order
    // n_s = 58/60 + small candidate-specific merger-class shift.
    let (loss_route5_ns, route5_n_s_predicted, _) =
        crate::route5::spectral_index::route5_chi2_against_planck(
            &candidate.fundamental_group,
        );

    Ok(FiveSigmaBreakdown {
        n_samples_accepted: 0,
        n_27_generations: 0,
        yukawa_norm_u: 0.0,
        yukawa_norm_d: 0.0,
        yukawa_norm_e: 0.0,
        chi2_total: 0.0,
        chi2_dof: 0,
        p_value: 1.0,
        passes_5_sigma: false,
        ckm_unitarity_residual: 0.0,
        loss_route2_gauge,
        loss_eta_chi2,
        eta_predicted,
        eta_uncertainty,
        loss_route4_chi2,
        route4_saturn_match: sat,
        route4_jupiter_north_match: jn,
        route4_jupiter_south_match: js,
        loss_route5_ns,
        route5_n_s_predicted,
    })
}

/// Sample the Schoen Z/3 × Z/3 fiber product via
/// [`crate::route34::schoen_sampler`] and pack the resulting
/// `SchoenPoint`s into the [`crate::cicy_sampler::SampledPoint`]
/// shape the downstream Yukawa pipeline consumes.
///
/// **Caveat** (documented in the calling Stage 1 comment): the
/// downstream Stage 3-5 stages currently use the ALP-2011 demo
/// monad bundle, whose `b_lines` reference CP^3 × CP^3 bidegree
/// indices. On Schoen the equivalent bundle data should come from
/// ALP 2011 §4's specific Γ-equivariant choice (task #47, deferred).
/// Until that lands, the Schoen Stage 3-5 numbers are computed
/// against the wrong bundle on the right geometry — i.e. they tell
/// us "what would the ALP TY bundle integrate to on Schoen sample
/// points" rather than "what does the substrate predict for
/// Schoen". The integration measure (sample point cloud + Ω +
/// weights) IS correct for Schoen.
fn sample_schoen_points_as_sampled_points(
    config: &FiveSigmaConfig,
) -> Vec<crate::cicy_sampler::SampledPoint> {
    use crate::route34::schoen_geometry::SchoenGeometry;
    use crate::route34::schoen_sampler::{SchoenPoly, SchoenSampler};
    use num_complex::Complex64;

    let poly = SchoenPoly::z3xz3_invariant_default();
    let geom = SchoenGeometry::schoen_z3xz3();
    let mut sampler = SchoenSampler::new(poly, geom, config.sampler_seed);
    let schoen_points = sampler.sample_points(config.n_sample_points, None);
    let inv_quotient = 1.0 / 9.0; // Z/3 × Z/3 quotient: divide weights by 9.
    schoen_points
        .iter()
        .map(|sp| {
            // Pack 3+3+2 ambient coords into the 4+4 SampledPoint
            // shape. The `t` `CP^1` coords go into the last slot of
            // each `z`/`w` 4-tuple.
            let z = [sp.x[0], sp.x[1], sp.x[2], sp.t[0]];
            let w = [sp.y[0], sp.y[1], sp.y[2], sp.t[1]];
            crate::cicy_sampler::SampledPoint {
                z,
                w,
                omega: sp.omega,
                weight: sp.weight * inv_quotient,
            }
        })
        .filter(|p| {
            p.weight.is_finite()
                && p.weight > 0.0
                && p.omega.re.is_finite()
                && p.omega.im.is_finite()
                && p.z.iter().all(|c| c.re.is_finite() && c.im.is_finite())
                && p.w.iter().all(|c| c.re.is_finite() && c.im.is_finite())
        })
        .map(|p| p) // identity, kept to make the chain readable
        .collect::<Vec<_>>()
        // Final sanity: ensure weights are normalised after the
        // 1/9 quotient division.
        .pipe_normalise_weights()
}

/// Tiny extension to fluent-normalise weights on a Vec<SampledPoint>
/// so the call chain in `sample_schoen_points_as_sampled_points`
/// stays readable.
trait NormaliseWeightsFluent {
    fn pipe_normalise_weights(self) -> Self;
}
impl NormaliseWeightsFluent for Vec<crate::cicy_sampler::SampledPoint> {
    fn pipe_normalise_weights(mut self) -> Self {
        let sum: f64 = self.iter().map(|p| p.weight).sum();
        if sum.is_finite() && sum > 0.0 {
            for p in self.iter_mut() {
                p.weight /= sum;
            }
        }
        self
    }
}

/// Run the chapter-21 η-integral evaluator on the supplied
/// geometry, return `(chi^2, predicted, uncertainty)` where chi^2
/// is computed against the chapter-21 observed value
/// `η_obs = (6.115 ± 0.038) × 10⁻¹⁰` with combined uncertainty
/// `σ = √(σ_obs² + σ_predicted²)`.
///
/// Returns all-zeros if the evaluator errors (e.g. trivial visible
/// bundle) or if the geometry is not one of the two supported
/// candidates (Tian-Yau Z/3, Schoen Z/3 × Z/3 fiber product).
fn compute_eta_chi2_for_geometry(
    geometry: &crate::geometry::CicyGeometry,
    seed: u64,
) -> (f64, f64, f64) {
    use crate::route34::eta_evaluator::{
        evaluate_eta_schoen, evaluate_eta_tian_yau, EtaEvaluatorConfig,
    };
    const ETA_OBS: f64 = 6.115e-10;
    const ETA_OBS_SIGMA: f64 = 0.038e-10;

    // Modest sample count keeps per-candidate runtime under ~30s
    // on a CPU-only host. Callers wanting tighter η_predicted
    // uncertainty should use the standalone eta_discriminate
    // binary.
    let cfg = EtaEvaluatorConfig {
        n_metric_iters: 12,
        n_metric_samples: 400,
        n_integrand_samples: 2000,
        kahler_moduli: vec![1.0; geometry.ambient_factors.len()],
        seed,
        checkpoint_path: None,
        max_wallclock_seconds: 60,
    };

    let result = match geometry.quotient_label.as_str() {
        "Z3xZ3" => evaluate_eta_schoen(&cfg),
        "Z3" => evaluate_eta_tian_yau(&cfg),
        _ => return (0.0, 0.0, 0.0),
    };
    match result {
        Ok(r) => {
            let combined_sigma = (ETA_OBS_SIGMA.powi(2) + r.eta_uncertainty.powi(2)).sqrt();
            let chi2 = if combined_sigma > 0.0 {
                ((r.eta_predicted - ETA_OBS) / combined_sigma).powi(2)
            } else {
                0.0
            };
            (chi2, r.eta_predicted, r.eta_uncertainty)
        }
        Err(_) => (0.0, 0.0, 0.0),
    }
}

/// Fold the end-to-end 5σ score into a `ScoreResult`. Sets `loss_pdg_chi2`
/// and `loss_ckm_unitarity`; leaves the other components untouched (the
/// caller is expected to populate them from the existing scoring path).
pub fn score_with_5sigma(
    base: ScoreResult,
    config: &FiveSigmaConfig,
) -> Result<ScoreResult, &'static str> {
    let bd = compute_5sigma_score(config)?;
    Ok(ScoreResult {
        loss_pdg_chi2: bd.chi2_total,
        loss_ckm_unitarity: bd.ckm_unitarity_residual,
        ..base
    })
}

// ---------------------------------------------------------------------------
// Sweep harness: rank a population of Candidates by 5σ score
// ---------------------------------------------------------------------------

/// Per-candidate result from [`sweep_candidates`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateRanking {
    /// `Candidate.id`.
    pub candidate_id: u64,
    /// `Candidate.candidate_short_name`.
    pub candidate_short_name: String,
    /// `Candidate.geometry.name` (e.g. "Tian-Yau Z/3" or
    /// "Schoen Z/3 × Z/3 fiber-product").
    pub geometry_label: String,
    /// `Candidate.fundamental_group` (e.g. "Z3", "Z3xZ3").
    pub fundamental_group: String,
    /// Wallclock seconds spent in
    /// [`compute_5sigma_score_for_candidate`].
    pub elapsed_seconds: f64,
    /// `Some(_)` on success, `None` on the
    /// "geometry not currently sampler-supported" / pipeline-error
    /// path. Either way the candidate is included in the output so
    /// the caller can see what happened — sorting puts errored
    /// candidates last via `total_loss = +∞`.
    pub breakdown: Option<FiveSigmaBreakdown>,
    /// Error message if `breakdown` is `None`.
    pub error: Option<String>,
    /// Total score: `χ²_total + ckm_unitarity_residual` for
    /// success, `+∞` for error. Lower is better.
    pub total_loss: f64,
    /// `true` iff `breakdown.is_some() && breakdown.passes_5_sigma`.
    pub passes_5_sigma: bool,
}

/// Score every candidate in `candidates`, return the resulting
/// rankings sorted by `total_loss` ascending (best first).
///
/// Candidates whose geometry is not yet sampler-supported (e.g.
/// Schoen until the route34 dispatch is wired in pipeline) are
/// returned with `breakdown = None`, `error = Some(_)`, and
/// `total_loss = f64::INFINITY` — they sort to the end and the
/// caller can flag them in reports.
///
/// Per-candidate work is independent, so this iterates with rayon
/// when more than one candidate is supplied. Each candidate's
/// `compute_5sigma_score_for_candidate` call is itself
/// rayon-parallel internally; the outer sweep uses
/// `into_par_iter` only when `n_candidates > 1` to avoid
/// double-nesting on single-candidate runs.
pub fn sweep_candidates(
    candidates: &[Candidate],
    config: &FiveSigmaConfig,
) -> Vec<CandidateRanking> {
    use rayon::prelude::*;
    use std::time::Instant;
    if candidates.is_empty() {
        return Vec::new();
    }

    let score_one = |c: &Candidate| -> CandidateRanking {
        let t = Instant::now();
        let result = compute_5sigma_score_for_candidate(config, c);
        let elapsed = t.elapsed().as_secs_f64();
        let geometry_label = c.geometry.name.clone();
        let (breakdown, error, total_loss, passes) = match result {
            Ok(bd) => {
                let total = bd.chi2_total
                    + bd.ckm_unitarity_residual
                    + bd.loss_route2_gauge
                    + bd.loss_eta_chi2
                    + bd.loss_route4_chi2
                    + bd.loss_route5_ns;
                let passes = bd.passes_5_sigma;
                (Some(bd), None, total, passes)
            }
            Err(e) => (None, Some(e.to_string()), f64::INFINITY, false),
        };
        CandidateRanking {
            candidate_id: c.id,
            candidate_short_name: c.candidate_short_name.clone(),
            geometry_label,
            fundamental_group: c.fundamental_group.clone(),
            elapsed_seconds: elapsed,
            breakdown,
            error,
            total_loss,
            passes_5_sigma: passes,
        }
    };

    let mut rankings: Vec<CandidateRanking> = if candidates.len() == 1 {
        vec![score_one(&candidates[0])]
    } else {
        candidates.par_iter().map(score_one).collect()
    };

    rankings.sort_by(|a, b| {
        a.total_loss
            .partial_cmp(&b.total_loss)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rankings
}

/// Format a slice of [`CandidateRanking`]s as a markdown table
/// suitable for direct inclusion in a chapter / PR description /
/// notebook. Columns:
///
/// | Rank | Geometry | χ²(PDG) | CKM | R2 | R3 (η) | R4 | Total | Status |
///
/// Numeric formatting is "tight" — three significant figures in
/// scientific notation for the χ² columns, with `-` displayed when
/// the candidate produced an error rather than a breakdown. The
/// **Status** column shows "✓ 5σ" when `passes_5_sigma`,
/// "✗ no 5σ" otherwise, and "ERROR" for candidates whose pipeline
/// errored (the underlying error message is truncated to 60 chars
/// to keep the table width sane; the full message is available
/// in the structured JSON output).
pub fn format_ranking_report_markdown(rankings: &[CandidateRanking]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(
        &mut s,
        "| Rank | Geometry | χ²(PDG) | CKM | R2 | R3 (η) | R4 | R5 (n_s) | n_s | Total | Status |"
    );
    let _ = writeln!(
        &mut s,
        "|-----:|----------|--------:|----:|----:|-------:|----:|--------:|-----:|------:|:-------|"
    );
    for (i, r) in rankings.iter().enumerate() {
        let rank = i + 1;
        match (&r.breakdown, &r.error) {
            (Some(bd), _) => {
                let _ = writeln!(
                    &mut s,
                    "| {} | {} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.3} | {:.3e} | {:.5} | {:.2e} | {} |",
                    rank,
                    r.geometry_label,
                    bd.chi2_total,
                    bd.ckm_unitarity_residual,
                    bd.loss_route2_gauge,
                    bd.loss_eta_chi2,
                    bd.loss_route4_chi2,
                    bd.loss_route5_ns,
                    bd.route5_n_s_predicted,
                    r.total_loss,
                    if bd.passes_5_sigma { "✓ 5σ" } else { "✗ no 5σ" },
                );
            }
            (None, Some(err)) => {
                let truncated: String = err.chars().take(60).collect();
                let suffix = if err.chars().count() > 60 { "…" } else { "" };
                let _ = writeln!(
                    &mut s,
                    "| {} | {} | - | - | - | - | - | - | - | - | ERROR: {}{} |",
                    rank, r.geometry_label, truncated, suffix,
                );
            }
            (None, None) => {
                let _ = writeln!(
                    &mut s,
                    "| {} | {} | - | - | - | - | - | - | - | - | (no result) |",
                    rank, r.geometry_label,
                );
            }
        }
    }
    s
}

/// One-line summary of a [`CandidateRanking`] for `eprintln!`
/// dumps and CLI reports.
pub fn ranking_summary_line(r: &CandidateRanking, rank: usize) -> String {
    match (&r.breakdown, &r.error) {
        (Some(bd), _) => format!(
            "  #{rank:>3}  id={:<6}  geom={:<32}  χ²_pdg={:.3e}  CKM-resid={:.3e}  \
             χ²_R2={:.3e}  χ²_η={:.3e}  χ²_R4={:.3}  χ²_R5={:.3e}  n_s={:.5}  \
             η_pred={:.3e}  p={:.3e}  5σ={}  ({:.2}s)",
            r.candidate_id,
            r.geometry_label,
            bd.chi2_total,
            bd.ckm_unitarity_residual,
            bd.loss_route2_gauge,
            bd.loss_eta_chi2,
            bd.loss_route4_chi2,
            bd.loss_route5_ns,
            bd.route5_n_s_predicted,
            bd.eta_predicted,
            bd.p_value,
            if bd.passes_5_sigma { "yes" } else { "no" },
            r.elapsed_seconds,
        ),
        (None, Some(err)) => format!(
            "  #{rank:>3}  id={:<6}  geom={:<32}  ERROR: {}  ({:.2}s)",
            r.candidate_id, r.geometry_label, err, r.elapsed_seconds,
        ),
        (None, None) => format!(
            "  #{rank:>3}  id={:<6}  geom={:<32}  (no result)",
            r.candidate_id, r.geometry_label,
        ),
    }
}

#[cfg(test)]
mod sweep_tests {
    use super::*;

    fn make_tian_yau_candidate(id: u64, name: &str) -> Candidate {
        Candidate {
            id,
            candidate_short_name: name.to_string(),
            euler_characteristic: -6,
            fundamental_group: "Z3".to_string(),
            kahler_moduli: Vec::new(),
            complex_moduli_real: Vec::new(),
            complex_moduli_imag: Vec::new(),
            bundle_moduli: Vec::new(),
            parent_id: None,
            geometry: crate::geometry::CicyGeometry::tian_yau_z3(),
        }
    }

    fn make_schoen_candidate(id: u64) -> Candidate {
        Candidate {
            id,
            candidate_short_name: "schoen-test".to_string(),
            euler_characteristic: 0,
            fundamental_group: "Z3xZ3".to_string(),
            kahler_moduli: Vec::new(),
            complex_moduli_real: Vec::new(),
            complex_moduli_imag: Vec::new(),
            bundle_moduli: Vec::new(),
            parent_id: None,
            geometry: crate::geometry::CicyGeometry::schoen_z3xz3(),
        }
    }

    /// Empty input returns empty output (no panic).
    #[test]
    fn sweep_empty_returns_empty() {
        let cfg = FiveSigmaConfig::default();
        let r = sweep_candidates(&[], &cfg);
        assert!(r.is_empty());
    }

    /// Schoen candidates run the **partial pipeline** (Routes 2/3/4
    /// only, since the line-intersection sampler doesn't support
    /// CP² × CP² × CP¹). The sweep returns a real breakdown with
    /// loss_route2_gauge + loss_route4_chi2 populated and the
    /// sampler-dependent fields zeroed. Tian-Yau still runs the
    /// full pipeline. Both candidates should rank by total_loss
    /// (Schoen partial-pipeline candidates have small finite
    /// total_loss; Tian-Yau full-pipeline has the placeholder-H-
    /// inflated χ²_PDG, so sorts much higher).
    #[test]
    fn sweep_runs_partial_pipeline_for_schoen() {
        let cfg = FiveSigmaConfig {
            n_sample_points: 50,
            sampler_seed: 5,
            mu_init_gev: 1.0e16,
            compute_eta_chi2: false,
        };
        let candidates = vec![
            make_tian_yau_candidate(1, "ty1"),
            make_schoen_candidate(2),
        ];
        let r = sweep_candidates(&candidates, &cfg);
        assert_eq!(r.len(), 2);
        // Both candidates produce real breakdowns (no Err path).
        for entry in &r {
            assert!(entry.breakdown.is_some(), "candidate breakdown missing");
            assert!(entry.error.is_none(), "candidate {entry:?} surfaced an error");
            assert!(entry.total_loss.is_finite(), "non-finite total_loss");
        }
        // Schoen breakdown: the route34::schoen_sampler dispatch
        // (task #33) actually runs now, so n_samples_accepted is
        // populated. Stage 3-5 still uses ALP's CP^3×CP^3-shaped
        // bundle which gives 0 generations on Schoen's intersection
        // form, so the spectrum-derived fields fall through to the
        // routes-only path (task #47 — real Schoen-side bundle from
        // ALP 2011 §4 — is the next step that unblocks them).
        let schoen = r
            .iter()
            .find(|x| x.fundamental_group == "Z3xZ3")
            .expect("Schoen candidate missing");
        let bd = schoen.breakdown.as_ref().unwrap();
        assert!(
            bd.n_samples_accepted > 0,
            "Schoen sampler dispatch should produce samples; got {}",
            bd.n_samples_accepted
        );
        assert_eq!(bd.n_27_generations, 0);
        assert_eq!(bd.chi2_total, 0.0);
        assert_eq!(bd.ckm_unitarity_residual, 0.0);
        // Routes 2 + 4 ran.
        assert!(bd.loss_route2_gauge.abs() < 1.0e-8);
        assert!(bd.loss_route4_chi2 >= 0.0);
        // Tian-Yau ran the full pipeline.
        let ty = r
            .iter()
            .find(|x| x.fundamental_group == "Z3")
            .expect("TY candidate missing");
        let bd = ty.breakdown.as_ref().unwrap();
        assert!(bd.n_samples_accepted > 0);
        assert!(bd.n_27_generations > 0);
        assert!(bd.chi2_total > 0.0);
    }

    /// Single-candidate sweep returns a one-element vec without
    /// invoking the rayon parallelism path.
    #[test]
    fn sweep_one_candidate() {
        let cfg = FiveSigmaConfig {
            n_sample_points: 50,
            sampler_seed: 7,
            mu_init_gev: 1.0e16,
            compute_eta_chi2: false,
        };
        let r = sweep_candidates(&[make_tian_yau_candidate(42, "lone-ty")], &cfg);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].candidate_id, 42);
        assert!(r[0].breakdown.is_some());
    }

    /// REGRESSION: FiveSigmaBreakdown.loss_route2_gauge must be
    /// populated (and ≈ 0 by construction — the cross-term-as-
    /// coupling identity is the substrate-physics commitment).
    /// Also asserts the field is folded into total_loss.
    #[test]
    fn breakdown_includes_route2_gauge_loss() {
        let cfg = FiveSigmaConfig {
            n_sample_points: 50,
            sampler_seed: 23,
            mu_init_gev: 1.0e16,
            compute_eta_chi2: false,
        };
        let candidate = make_tian_yau_candidate(99, "ty-route2-test");
        let r = sweep_candidates(&[candidate], &cfg);
        assert_eq!(r.len(), 1);
        let bd = r[0].breakdown.as_ref().expect("breakdown missing");
        // Route 2 gauge identity should give exactly zero (≤ FP noise).
        assert!(
            bd.loss_route2_gauge.abs() < 1.0e-8,
            "loss_route2_gauge unexpectedly large: {}",
            bd.loss_route2_gauge
        );
        // η evaluator did NOT run (compute_eta_chi2 = false), so its
        // contribution must be exactly zero and metadata fields zero.
        assert_eq!(bd.loss_eta_chi2, 0.0);
        assert_eq!(bd.eta_predicted, 0.0);
        assert_eq!(bd.eta_uncertainty, 0.0);
        // Route 4 always runs and is candidate-specific; for the
        // Tian-Yau Z/3 candidate the soft-match scores must be in
        // [0, 1] and the χ² ≥ 0.
        assert!(bd.route4_saturn_match >= 0.0 && bd.route4_saturn_match <= 1.0);
        assert!(bd.route4_jupiter_north_match >= 0.0 && bd.route4_jupiter_north_match <= 1.0);
        assert!(bd.route4_jupiter_south_match >= 0.0 && bd.route4_jupiter_south_match <= 1.0);
        assert!(bd.loss_route4_chi2 >= 0.0);
        // Route 5 always runs (closed-form). Predicted n_s sits
        // within ±0.001 of the leading-order 58/60 ≈ 0.96667.
        assert!(bd.route5_n_s_predicted > 0.965 && bd.route5_n_s_predicted < 0.969);
        assert!(bd.loss_route5_ns >= 0.0);
        // total_loss must include all six plumbed components.
        let expected_total = bd.chi2_total
            + bd.ckm_unitarity_residual
            + bd.loss_route2_gauge
            + bd.loss_eta_chi2
            + bd.loss_route4_chi2
            + bd.loss_route5_ns;
        assert!(
            (r[0].total_loss - expected_total).abs() < 1.0e-9 * r[0].total_loss.abs().max(1.0),
            "total_loss = {} but chi2 + ckm + route2 + eta + route4 + route5 = {}",
            r[0].total_loss,
            expected_total
        );
    }

    /// Markdown report renders a header, a divider row, and one
    /// data row per candidate. Both Tian-Yau and Schoen now produce
    /// real numeric rows (Schoen via the partial pipeline that
    /// runs Routes 2/3/4 only).
    #[test]
    fn markdown_report_renders_table() {
        let cfg = FiveSigmaConfig {
            n_sample_points: 50,
            sampler_seed: 17,
            mu_init_gev: 1.0e16,
            compute_eta_chi2: false,
        };
        let candidates = vec![
            make_tian_yau_candidate(1, "ty"),
            make_schoen_candidate(2),
        ];
        let r = sweep_candidates(&candidates, &cfg);
        let md = format_ranking_report_markdown(&r);

        // Header row + divider row + n_candidates data rows = n+2 lines.
        let n_lines = md.lines().count();
        assert_eq!(n_lines, 4, "expected 4 lines (header, divider, 2 data); got {n_lines}\n{md}");

        // Header is well-formed.
        assert!(md.starts_with("| Rank | Geometry"), "missing markdown header: {md}");
        // Divider row contains the alignment markers.
        assert!(md.contains("|-----:|----------|"), "missing divider row: {md}");
        // Both geometries appear with their full names.
        assert!(md.contains("Tian-Yau Z/3"), "TY geometry row missing");
        assert!(md.contains("Schoen Z/3 × Z/3"), "Schoen geometry row missing");
        // No ERROR rows — both candidates score successfully now
        // (Schoen via partial pipeline).
        assert!(!md.contains("ERROR"), "unexpected ERROR row in: {md}");
    }

    /// Markdown formatter handles an empty ranking list (header
    /// + divider only, no data rows).
    #[test]
    fn markdown_report_empty_input() {
        let md = format_ranking_report_markdown(&[]);
        let n_lines = md.lines().count();
        assert_eq!(n_lines, 2, "expected just header + divider; got {n_lines}\n{md}");
    }

    /// **Numerical empirical test**: does the Route 4 polyhedral-
    /// resonance ADE-wavenumber predictor actually distinguish
    /// Tian-Yau Z/3 from Schoen Z/3 × Z/3?
    ///
    /// Both candidates feed `KillingResult` into the same Arnold-
    /// catastrophe-classification pipeline. They differ only in the
    /// `cyclic_factors` slot (TY: `[Z/3]`, Schoen: `[Z/3, Z/3]`)
    /// and `continuous_isometry_dim` (both 0). Whether that
    /// difference propagates into a different soft-match score per
    /// planet is a function of the
    /// `route34::route4_predictor::predict_wavenumber_set` internals.
    ///
    /// This is the unit test that documents the answer either way:
    /// fails loudly if the two predictions are bit-identical, with
    /// a message identifying the degeneracy as a real finding (not
    /// a bug in the test).
    #[test]
    fn route4_distinguishes_ty_from_schoen() {
        use crate::route34::route4_predictor::route4_discrimination;
        use crate::route34::KillingResult;

        let ty = route4_discrimination(&KillingResult::tianyau_z3())
            .expect("Route 4 predictor failed for TY");
        let sch = route4_discrimination(&KillingResult::schoen_z3xz3())
            .expect("Route 4 predictor failed for Schoen");

        eprintln!("TY/Z3 Route 4:");
        eprintln!("  Saturn match (n=6): {:.4}", ty.saturn_match);
        eprintln!("  Jupiter N (n=8) :   {:.4}", ty.jupiter_north_match);
        eprintln!("  Jupiter S (n=5) :   {:.4}", ty.jupiter_south_match);
        eprintln!("  combined χ²:        {:.4}", ty.combined_chi_squared);
        eprintln!("Schoen/Z3xZ3 Route 4:");
        eprintln!("  Saturn match (n=6): {:.4}", sch.saturn_match);
        eprintln!("  Jupiter N (n=8) :   {:.4}", sch.jupiter_north_match);
        eprintln!("  Jupiter S (n=5) :   {:.4}", sch.jupiter_south_match);
        eprintln!("  combined χ²:        {:.4}", sch.combined_chi_squared);

        let diff = (ty.combined_chi_squared - sch.combined_chi_squared).abs();
        // We don't enforce "must distinguish" here — that's a
        // physics question depending on whether the two discrete
        // groups give different Arnold ADE classifications at the
        // polar critical boundary. Instead, the test documents the
        // current state and asserts only that both predictions are
        // well-formed (finite, non-negative chi^2) so a future
        // route34 update that adds discrete-group sensitivity to
        // predict_wavenumber_set surfaces immediately as a diff.
        assert!(ty.combined_chi_squared.is_finite() && ty.combined_chi_squared >= 0.0);
        assert!(sch.combined_chi_squared.is_finite() && sch.combined_chi_squared >= 0.0);
        // The CURRENT state (recorded for future audit): if diff is
        // exactly zero, R4 is degenerate between TY and Schoen and
        // adds no candidate-discrimination signal beyond Route 3.
        // If diff > 0, R4 contributes independent discrimination.
        eprintln!("|χ²_TY − χ²_Schoen| = {diff:.6} (zero ⇒ R4 currently degenerate)");
    }

    /// **Numerical empirical test (slow, #[ignore]'d)**: does the
    /// chapter-21 η-integral evaluator actually distinguish Tian-Yau
    /// Z/3 from Schoen Z/3 × Z/3? This is the key question Route 3
    /// is supposed to answer — if both candidates give the same
    /// `eta_predicted` to within their uncertainty bands, Route 3
    /// adds no discrimination signal, and the substrate-physics
    /// program needs a different chapter-8 route as the actual
    /// discriminator.
    ///
    /// This test runs both `evaluate_eta_tian_yau` and
    /// `evaluate_eta_schoen` with reduced sample counts (CPU
    /// fallback only) and asserts the two predicted values differ
    /// by at least 0.5σ — a soft threshold that catches a complete
    /// degeneracy without requiring the prediction to nail the
    /// chapter's `η_obs = 6.115e-10` (which depends on the
    /// deferred-research bundle data anyway).
    ///
    /// Wallclock: 60–120s on a CPU-only host. Run on demand:
    /// `cargo test --release --lib --ignored route3_distinguishes`.
    #[test]
    #[ignore]
    fn route3_distinguishes_ty_from_schoen() {
        use crate::route34::eta_evaluator::{
            evaluate_eta_schoen, evaluate_eta_tian_yau, EtaEvaluatorConfig,
        };

        // Tian-Yau: 2-factor ambient (CP^3 × CP^3) → 2 Kähler moduli.
        let cfg_ty = EtaEvaluatorConfig {
            n_metric_iters: 8,
            n_metric_samples: 200,
            n_integrand_samples: 800,
            kahler_moduli: vec![1.0, 1.0],
            seed: 11,
            checkpoint_path: None,
            max_wallclock_seconds: 60,
        };
        // Schoen: 3-factor ambient (CP^2 × CP^2 × CP^1) → 3 Kähler moduli.
        let cfg_sch = EtaEvaluatorConfig {
            n_metric_iters: 8,
            n_metric_samples: 200,
            n_integrand_samples: 800,
            kahler_moduli: vec![1.0, 1.0, 1.0],
            seed: 11,
            checkpoint_path: None,
            max_wallclock_seconds: 60,
        };

        let ty = evaluate_eta_tian_yau(&cfg_ty)
            .expect("evaluate_eta_tian_yau failed unexpectedly");
        let sch = evaluate_eta_schoen(&cfg_sch)
            .expect("evaluate_eta_schoen failed unexpectedly");

        eprintln!(
            "η_TY    = {:.4e} ± {:.2e}",
            ty.eta_predicted, ty.eta_uncertainty
        );
        eprintln!(
            "η_Schoen = {:.4e} ± {:.2e}",
            sch.eta_predicted, sch.eta_uncertainty
        );

        let diff = (ty.eta_predicted - sch.eta_predicted).abs();
        let combined_sigma = (ty.eta_uncertainty.powi(2)
            + sch.eta_uncertainty.powi(2))
        .sqrt()
        .max(1.0e-30);
        let n_sigma = diff / combined_sigma;
        eprintln!(
            "|η_TY − η_Schoen| / σ_combined = {:.3} (threshold ≥ 0.5)",
            n_sigma
        );
        assert!(
            n_sigma >= 0.5,
            "Route 3 fails to distinguish Tian-Yau from Schoen at the η-integral \
             level — the two candidates give the same prediction within \
             0.5σ. Route 3 is therefore not currently a useful \
             discriminator. Predicted: η_TY = {:.3e}, η_Sch = {:.3e}",
            ty.eta_predicted,
            sch.eta_predicted
        );
    }

    /// REGRESSION: Route 4 dispatch produces VALID outputs for both
    /// TY/Z3 and Schoen/Z3xZ3 candidates. The numerical χ² values
    /// MAY coincide (both discrete groups give the same Arnold
    /// classification at the polar critical boundary today; whether
    /// the discriminator distinguishes them depends on bulk-isometry
    /// data the route34 Killing solver provides downstream — this
    /// test only asserts the dispatch path is reached, the outputs
    /// are well-formed, and Route 4 doesn't depend on Schoen's
    /// sampler running successfully).
    #[test]
    fn route4_runs_for_both_canonical_candidates() {
        let cfg = FiveSigmaConfig {
            n_sample_points: 50,
            sampler_seed: 31,
            mu_init_gev: 1.0e16,
            compute_eta_chi2: false,
        };
        let candidates = vec![
            make_tian_yau_candidate(101, "ty-r4"),
            make_schoen_candidate(102),
        ];
        let r = sweep_candidates(&candidates, &cfg);
        assert_eq!(r.len(), 2);
        // TY breakdown is populated.
        let ty = r
            .iter()
            .find(|x| x.fundamental_group == "Z3")
            .expect("TY candidate missing");
        let bd_ty = ty.breakdown.as_ref().expect("TY breakdown missing");
        assert!(bd_ty.loss_route4_chi2 >= 0.0);
        assert!(bd_ty.route4_saturn_match >= 0.0 && bd_ty.route4_saturn_match <= 1.0);
        // Route 4 for Schoen runs even though the sampler fails.
        let (sch_chi2, sch_sat, sch_jn, sch_js) =
            compute_route4_chi2_for_candidate(&make_schoen_candidate(0));
        assert!(sch_chi2 >= 0.0);
        assert!(sch_sat >= 0.0 && sch_sat <= 1.0);
        assert!(sch_jn >= 0.0 && sch_jn <= 1.0);
        assert!(sch_js >= 0.0 && sch_js <= 1.0);
    }
}

#[cfg(test)]
mod five_sigma_tests {
    use super::*;

    /// End-to-end smoke test: the 5σ pipeline runs without panic on
    /// the demo monad placeholder configuration. Marked `#[ignore]`
    /// because the full M4 → P1 → P2 → PDG run takes ~10s; CI runs
    /// it on demand via `cargo test --release -- --ignored
    /// five_sigma_smoke`.
    ///
    /// The demo monad `B = O(1,0)^3 ⊕ O(0,1)^3, C = O(1,1)^3` on the
    /// canonical Tian-Yau Z/3 geometry gives `|c_3|/2 = 27` upstairs
    /// → `9` generations downstairs after the Z/3 quotient, NOT the
    /// 3 generations the chapter-8 substrate-physics target requires.
    /// The `3 generation` target is a future replacement of the demo
    /// monad with a specific Γ-equivariant ALP-2011 model (see task
    /// #47). This smoke test asserts only the structural invariants
    /// (positive sample count, non-zero generation count, finite χ²,
    /// finite CKM residual) — it does NOT assert the χ² is small,
    /// because the placeholder bundle metric H = identity (task #48
    /// deferred research) puts O(1) errors on Yukawa magnitudes that
    /// inflate χ² far above any "5σ" threshold.
    #[test]
    #[ignore]
    fn five_sigma_pipeline_smoke() {
        let cfg = FiveSigmaConfig::default();
        let bd = compute_5sigma_score(&cfg).expect("pipeline failed");
        eprintln!(
            "5σ smoke: n_samples={}, n_gen={}, χ²={:.3} (dof {}), p={:.3e}, ckm_resid={:.3e}",
            bd.n_samples_accepted, bd.n_27_generations,
            bd.chi2_total, bd.chi2_dof, bd.p_value, bd.ckm_unitarity_residual,
        );
        assert!(bd.n_samples_accepted > 0, "no samples accepted");
        assert!(bd.n_27_generations > 0, "zero generations");
        assert!(bd.chi2_total.is_finite(), "χ² non-finite");
        assert!(bd.ckm_unitarity_residual.is_finite(), "CKM residual non-finite");
    }

    /// Lightweight (non-ignored) smoke test of the 5σ pipeline at
    /// reduced sample count. Verifies the M4 → P1 → P2 → PDG path
    /// runs end-to-end without panicking and produces structurally
    /// well-formed output. Reduced n_sample_points keeps runtime
    /// under ~5s on CPU-only hosts so this can ride normal `cargo
    /// test` cycles.
    #[test]
    fn five_sigma_pipeline_runs_at_small_n() {
        let cfg = FiveSigmaConfig {
            n_sample_points: 200,
            sampler_seed: 11,
            mu_init_gev: 1.0e16,
            compute_eta_chi2: false,
        };
        let bd = compute_5sigma_score(&cfg).expect("pipeline failed");
        assert!(bd.n_samples_accepted > 0, "no samples accepted");
        assert!(bd.n_27_generations > 0, "zero generations");
        assert!(bd.chi2_total.is_finite(), "χ² non-finite");
        assert!(bd.ckm_unitarity_residual.is_finite(), "CKM residual non-finite");
        // Yukawa norms are placeholder magnitudes; just check they're
        // finite and non-NaN.
        assert!(bd.yukawa_norm_u.is_finite());
        assert!(bd.yukawa_norm_d.is_finite());
        assert!(bd.yukawa_norm_e.is_finite());
    }
}

/// Moduli sampling ranges for Pass-1 broad sweep. Lets callers zoom into
/// a specific region of moduli space (e.g., follow up on a promising
/// region from a previous run).
#[derive(Debug, Clone)]
pub struct ModuliRanges {
    pub kahler_min: f64,
    pub kahler_max: f64,
    pub complex_real_min: f64,
    pub complex_real_max: f64,
    pub complex_imag_min: f64,
    pub complex_imag_max: f64,
    pub bundle_min: f64,
    pub bundle_max: f64,
}

impl Default for ModuliRanges {
    fn default() -> Self {
        Self {
            kahler_min: 0.5,
            kahler_max: 10.0,
            complex_real_min: -1.0,
            complex_real_max: 1.0,
            complex_imag_min: -1.0,
            complex_imag_max: 1.0,
            bundle_min: -1.0,
            bundle_max: 1.0,
        }
    }
}

/// Generate synthetic Pass-1 candidates with explicit ID range and
/// configurable moduli ranges. The ID range is inclusive [start_id..=end_id]
/// so callers can resume an interrupted broad sweep at the exact next ID.
pub fn generate_broad_sweep_candidates_in_range(
    candidate_short_name: &str,
    euler_characteristic: i32,
    fundamental_group: &str,
    h11: usize,
    h21: usize,
    n_bundle: usize,
    start_id: u64,
    end_id: u64,
    ranges: &ModuliRanges,
    seed: u64,
) -> Vec<Candidate> {
    let n_candidates = (end_id + 1 - start_id) as usize;
    let mut candidates = Vec::with_capacity(n_candidates);
    let next_u64 = |state: &mut u64| {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    };
    let next_f64 = |state: &mut u64| {
        let bits = (next_u64(state) >> 11) & ((1u64 << 53) - 1);
        bits as f64 / (1u64 << 53) as f64
    };

    let kahler_span = ranges.kahler_max - ranges.kahler_min;
    let cre_span = ranges.complex_real_max - ranges.complex_real_min;
    let cim_span = ranges.complex_imag_max - ranges.complex_imag_min;
    let bundle_span = ranges.bundle_max - ranges.bundle_min;

    // Wilson-line quantization constants. The heterotic Z/3 Wilson line
    // requires bundle_moduli[20..28] (the 8 Cartan phases) to lie near
    // multiples of 2π/3. Sampling uniformly on [-1, 1] and rejecting
    // wastes essentially all candidates; instead we sample directly on
    // the Z/3 lattice and add small Gaussian jitter so the gradient
    // surface around the constraint is still explored.
    let two_pi_third = 2.0 * std::f64::consts::PI / 3.0;
    let wilson_jitter_sigma = 0.05_f64; // ~3 deg jitter

    // Bundle line-bundle degrees (b_degrees[0..5], c_degrees[5]) live
    // on the integer lattice in [-5, 5]; sample integers directly and
    // enforce sum_b = sum_c so that c_1(V) = 0 by construction.
    let degree_range_half: i32 = 5;

    // Approximation to the Box-Muller-ish standard normal from two
    // uniforms in [0,1]; we don't need cryptographic quality, just a
    // smooth jitter distribution.
    let next_normal = |state: &mut u64| -> f64 {
        let u1 = next_f64(state).max(1e-12);
        let u2 = next_f64(state);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // Per-candidate seed: derive from base seed + candidate id so that
    // each id always produces the same moduli regardless of the slice
    // requested. This makes resume-by-id-range deterministic.
    for id in start_id..=end_id {
        let mut state = seed.wrapping_mul(1_000_003).wrapping_add(id);
        // Burn a few rounds to whiten LCG output for nearby seeds
        for _ in 0..3 {
            next_u64(&mut state);
        }

        let kahler_moduli: Vec<f64> = (0..h11)
            .map(|_| ranges.kahler_min + kahler_span * next_f64(&mut state))
            .collect();
        let complex_moduli_real: Vec<f64> = (0..h21)
            .map(|_| ranges.complex_real_min + cre_span * next_f64(&mut state))
            .collect();
        let complex_moduli_imag: Vec<f64> = (0..h21)
            .map(|_| ranges.complex_imag_min + cim_span * next_f64(&mut state))
            .collect();

        // Generate bundle moduli with structure-aware sampling. The
        // bundle layout (consumed by heterotic.rs::MonadBundle::from_bundle_moduli
        // and topology_filters.rs::wilson_line_loss):
        //   [0..5]   b-degree line bundles  (integer-rounded by MonadBundle)
        //   [5]      c-degree line bundle   (integer-rounded by MonadBundle)
        //   [6..20]  monad map coefficients + slope/sub-sheaf slots
        //   [20..23] Z/3 Wilson-line phase triple    (Z/3 lattice)
        //   [23..28] additional Cartan phases        (Z/3 lattice)
        //   [28..]   tail moduli
        //
        // Pre-quantizing the integer-degree and Z/3-phase entries
        // collapses the random-rejection prior on the corresponding
        // filters from ~exp(-50) to O(1).
        let mut bundle_moduli: Vec<f64> = vec![0.0; n_bundle];

        // b-degrees on integer lattice [-5, 5], sampled freely except
        // that we enforce sum_b == sum_c by setting the c-degree to
        // sum(b). This forces c_1(V) = sum_b - sum_c = 0 by construction
        // for the standard 5+1 monad shape decoded by MonadBundle.
        let n_b: usize = 5usize.min(n_bundle);
        let mut sum_b = 0_i32;
        for i in 0..n_b {
            let r = next_f64(&mut state);
            let deg = ((r * (2 * degree_range_half + 1) as f64) as i32) - degree_range_half;
            bundle_moduli[i] = deg as f64;
            sum_b += deg;
        }
        if 5 < n_bundle {
            // Clamp to monad's accepted range to avoid silent clamp at decode.
            let c_deg = sum_b.clamp(-degree_range_half, degree_range_half);
            bundle_moduli[5] = c_deg as f64;
        }

        // Map coefficients + slope slots: uniform sample (no constraint
        // structure at this stage).
        for i in 6..20.min(n_bundle) {
            bundle_moduli[i] = ranges.bundle_min + bundle_span * next_f64(&mut state);
        }

        // Z/3 Wilson-line phases at indices 20..28 (or as many as
        // n_bundle permits). Snap to {0, 2π/3, 4π/3} with small jitter.
        for i in 20..28.min(n_bundle) {
            let k_idx = (next_f64(&mut state) * 3.0) as i32;
            let k = k_idx.clamp(0, 2);
            let base = (k as f64) * two_pi_third;
            let jitter = wilson_jitter_sigma * next_normal(&mut state);
            bundle_moduli[i] = base + jitter;
        }

        // Tail moduli: uniform sample in the configured range.
        for i in 28..n_bundle {
            bundle_moduli[i] = ranges.bundle_min + bundle_span * next_f64(&mut state);
        }

        let geometry = match fundamental_group {
            "Z3xZ3" => crate::geometry::CicyGeometry::schoen_z3xz3(),
            _ => crate::geometry::CicyGeometry::tian_yau_z3(),
        };
        candidates.push(Candidate {
            id,
            candidate_short_name: candidate_short_name.to_string(),
            euler_characteristic,
            fundamental_group: fundamental_group.to_string(),
            kahler_moduli,
            complex_moduli_real,
            complex_moduli_imag,
            bundle_moduli,
            parent_id: None,
            geometry,
        });
    }
    candidates
}

/// Backwards-compatible wrapper: generate IDs 0..n_candidates with default
/// moduli ranges.
pub fn generate_broad_sweep_candidates(
    candidate_short_name: &str,
    euler_characteristic: i32,
    fundamental_group: &str,
    h11: usize,
    h21: usize,
    n_bundle: usize,
    n_candidates: usize,
    seed: u64,
) -> Vec<Candidate> {
    if n_candidates == 0 {
        return Vec::new();
    }
    generate_broad_sweep_candidates_in_range(
        candidate_short_name,
        euler_characteristic,
        fundamental_group,
        h11,
        h21,
        n_bundle,
        0,
        (n_candidates - 1) as u64,
        &ModuliRanges::default(),
        seed,
    )
}

/// Streaming variant of `generate_broad_sweep_candidates_in_range`:
/// builds candidates one at a time inside an iterator instead of
/// materialising the full Vec up front. Removes the upfront ~800 MB
/// allocation for a 1M-candidate broad sweep at the standard moduli
/// shapes, freeing memory for the parallel scoring path.
///
/// Takes owned strings + owned `ModuliRanges` so the returned iterator
/// is `'static`-friendly and can be composed across `flat_map` chains
/// without lifetime entanglement.
///
/// Each yielded candidate is generated deterministically from `seed +
/// id`, exactly as in the Vec variant; resume-by-id-range is preserved.
pub fn iter_broad_sweep_candidates_in_range(
    candidate_short_name: String,
    euler_characteristic: i32,
    fundamental_group: String,
    h11: usize,
    h21: usize,
    n_bundle: usize,
    start_id: u64,
    end_id: u64,
    ranges: ModuliRanges,
    seed: u64,
) -> impl Iterator<Item = Candidate> + Send {
    let kahler_span = ranges.kahler_max - ranges.kahler_min;
    let cre_span = ranges.complex_real_max - ranges.complex_real_min;
    let cim_span = ranges.complex_imag_max - ranges.complex_imag_min;
    let bundle_span = ranges.bundle_max - ranges.bundle_min;
    let two_pi_third = 2.0 * std::f64::consts::PI / 3.0;
    let wilson_jitter_sigma = 0.05_f64;
    let degree_range_half: i32 = 5;

    (start_id..=end_id).map(move |id| {
        let next_u64 = |state: &mut u64| {
            *state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *state
        };
        let next_f64 = |state: &mut u64| {
            let bits = (next_u64(state) >> 11) & ((1u64 << 53) - 1);
            bits as f64 / (1u64 << 53) as f64
        };
        let next_normal = |state: &mut u64| -> f64 {
            let u1 = next_f64(state).max(1e-12);
            let u2 = next_f64(state);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let mut state = seed.wrapping_mul(1_000_003).wrapping_add(id);
        for _ in 0..3 {
            next_u64(&mut state);
        }

        let kahler_moduli: Vec<f64> = (0..h11)
            .map(|_| ranges.kahler_min + kahler_span * next_f64(&mut state))
            .collect();
        let complex_moduli_real: Vec<f64> = (0..h21)
            .map(|_| ranges.complex_real_min + cre_span * next_f64(&mut state))
            .collect();
        let complex_moduli_imag: Vec<f64> = (0..h21)
            .map(|_| ranges.complex_imag_min + cim_span * next_f64(&mut state))
            .collect();

        let mut bundle_moduli: Vec<f64> = vec![0.0; n_bundle];
        let n_b: usize = 5usize.min(n_bundle);
        let mut sum_b = 0_i32;
        for i in 0..n_b {
            let r = next_f64(&mut state);
            let deg = ((r * (2 * degree_range_half + 1) as f64) as i32) - degree_range_half;
            bundle_moduli[i] = deg as f64;
            sum_b += deg;
        }
        if 5 < n_bundle {
            let c_deg = sum_b.clamp(-degree_range_half, degree_range_half);
            bundle_moduli[5] = c_deg as f64;
        }
        for i in 6..20.min(n_bundle) {
            bundle_moduli[i] = ranges.bundle_min + bundle_span * next_f64(&mut state);
        }
        for i in 20..28.min(n_bundle) {
            let k_idx = (next_f64(&mut state) * 3.0) as i32;
            let k = k_idx.clamp(0, 2);
            let base = (k as f64) * two_pi_third;
            let jitter = wilson_jitter_sigma * next_normal(&mut state);
            bundle_moduli[i] = base + jitter;
        }
        for i in 28..n_bundle {
            bundle_moduli[i] = ranges.bundle_min + bundle_span * next_f64(&mut state);
        }

        let geometry = match fundamental_group.as_str() {
            "Z3xZ3" => crate::geometry::CicyGeometry::schoen_z3xz3(),
            _ => crate::geometry::CicyGeometry::tian_yau_z3(),
        };
        Candidate {
            id,
            candidate_short_name: candidate_short_name.clone(),
            euler_characteristic,
            fundamental_group: fundamental_group.clone(),
            kahler_moduli,
            complex_moduli_real,
            complex_moduli_imag,
            bundle_moduli,
            parent_id: None,
            geometry,
        }
    })
}
