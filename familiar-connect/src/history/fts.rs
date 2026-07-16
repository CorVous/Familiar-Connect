//! Tantivy full-text index + `familiar_en` analyzer (subsystem 03; Python
//! `history/fts.py`).
//!
//! One [`TantivyFts`] per indexed relational table (`turns`, `facts`). The index
//! lives OUTSIDE the SQLite database — on disk under `<db_dir>/fts/<name>/`, or
//! fully in RAM when constructed with [`TantivyFts::in_memory`] (tests /
//! `:memory:` stores). Documents are `(row_id: i64 stored/indexed/fast,
//! content: text unstored)`; upserts are delete-by-row_id then add.
//!
//! The analyzer chain matches Python's tantivy pipeline **exactly** so BM25
//! rankings are identical (spec 03 behavior 17): simple tokenizer →
//! remove_long(64) → lowercase → ascii_fold → the custom stopwords →
//! English stemmer. tantivy is native Rust here, so this is the same crate the
//! Python bindings wrap.
//!
//! ## Commit-retry seam (behavior 22, DESIGN §4.8)
//!
//! `commit()` occasionally races a Windows antivirus segment-scan that briefly
//! locks a freshly-written `.term` file, surfacing as a `PermissionDenied`
//! I/O error. Those transient-lock errors are retried with short backoffs
//! (`50ms`, `200ms`, `500ms`) then one final propagating attempt; every other
//! error fails fast. [`TantivyFts::set_commit_fault`] injects a fake failing
//! commit for the three retry tests (Python monkeypatched `_commit_writer`);
//! [`TantivyFts::set_retry_delays`] collapses the backoffs so those tests run
//! fast.

use std::io::ErrorKind;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use regex::Regex;
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::QueryParser;
use tantivy::schema::{
    FAST, Field, INDEXED, IndexRecordOption, STORED, Schema, TextFieldIndexing, TextOptions,
    Value as _,
};
use tantivy::tokenizer::{
    AsciiFoldingFilter, Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, Stemmer,
    StopWordFilter, TextAnalyzer,
};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};

use super::StoreError;
use super::store::FtsIndex;

/// Backoff delays before re-attempting a transient-lock commit failure. Seconds:
/// total worst-case wait ≈ 0.75s before the final propagating attempt.
const COMMIT_RETRY_DELAYS: [Duration; 3] = [
    Duration::from_millis(50),
    Duration::from_millis(200),
    Duration::from_millis(500),
];

/// Substrings that mark a tantivy commit error as a transient Windows AV file
/// lock (worth retrying). Matched against the error's string form because the
/// Rust `IoError` Debug repr can drift between tantivy versions.
const LOCK_SIGNATURES: [&str; 3] = ["PermissionDenied", "Access is denied", "os error 5"];

/// Registered analyzer name (shared by both the index-time tokenizer and the
/// query parser).
const ANALYZER_NAME: &str = "familiar_en";

/// Writer heap (bytes) and thread count — matches Python `writer(heap_size=…,
/// num_threads=1)`. 15 MB is tantivy's per-thread minimum.
const WRITER_HEAP_BYTES: usize = 15_000_000;

/// Common English stopwords dropped before FTS matching.
///
/// Verbatim copy of `_FTS_STOPWORDS` in `fts.py` (the actual list is 87 words,
/// not the 88 the spec quotes; includes chat fillers `hey`, `hi`, `lol`, `ok`,
/// `know`, `yes`). Changing this shifts BM25 rankings.
const FTS_STOPWORDS: [&str; 87] = [
    "a", "about", "an", "and", "any", "anything", "are", "as", "at", "be", "been", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "for", "from", "had", "has", "have",
    "having", "he", "her", "here", "hey", "hi", "him", "his", "how", "i", "if", "in", "is", "it",
    "its", "just", "know", "lol", "me", "my", "no", "not", "of", "ok", "on", "or", "our", "out",
    "she", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "to", "too", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "yes", "you", "your", "yours",
];

/// Test seam consulted before every real writer commit.
///
/// Returns `Some(err)` to simulate a commit failure this attempt, or `None` to
/// let the real commit proceed. Mirrors Python's monkeypatched `_commit_writer`.
pub type CommitFault = Box<dyn FnMut() -> Option<std::io::Error> + Send>;

/// Strip tantivy query syntax before parse: any run of non-word, non-space
/// characters becomes a single space, then lowercase. `\w` is Unicode-aware
/// (regex crate default), matching Python's `re.compile(r"[^\w\s]+")`.
fn sanitize_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[^\w\s]+").expect("static FTS sanitize regex is valid"))
}

fn sanitize_query(query: &str) -> String {
    sanitize_regex().replace_all(query, " ").to_lowercase()
}

/// Build the `familiar_en` analyzer. Filter order is load-bearing (BM25 parity):
/// remove_long(64) → lowercase → ascii_fold → custom stopwords → English stem.
fn build_analyzer() -> TextAnalyzer {
    TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(64))
        .filter(LowerCaser)
        .filter(AsciiFoldingFilter)
        .filter(StopWordFilter::remove(
            FTS_STOPWORDS.iter().map(|&w| w.to_owned()),
        ))
        .filter(Stemmer::new(Language::English))
        .build()
}

/// Schema: `row_id` (i64 stored/indexed/fast) + `content` (text, unstored,
/// tokenized by `familiar_en` with freqs+positions).
fn build_schema() -> Schema {
    let mut sb = Schema::builder();
    sb.add_i64_field("row_id", INDEXED | STORED | FAST);
    let content_indexing = TextFieldIndexing::default()
        .set_tokenizer(ANALYZER_NAME)
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let content_options = TextOptions::default().set_indexing_options(content_indexing);
    sb.add_text_field("content", content_options);
    sb.build()
}

/// A commit attempt's failure, retained long enough to classify it as a
/// retryable transient lock or a fail-fast error.
enum CommitErr {
    /// Test-injected failure.
    Injected(std::io::Error),
    /// A real tantivy commit error.
    Tantivy(tantivy::TantivyError),
}

impl CommitErr {
    fn is_transient_lock(&self) -> bool {
        match self {
            Self::Injected(io) => {
                io.kind() == ErrorKind::PermissionDenied || string_is_lock(&io.to_string())
            }
            Self::Tantivy(err) => string_is_lock(&err.to_string()),
        }
    }

    fn into_store_error(self) -> StoreError {
        match self {
            Self::Injected(io) => StoreError::Fts(io.to_string()),
            Self::Tantivy(err) => StoreError::Fts(err.to_string()),
        }
    }
}

fn string_is_lock(msg: &str) -> bool {
    LOCK_SIGNATURES.iter().any(|sig| msg.contains(sig))
}

/// Tantivy index over `(row_id, content)` for one relational table.
///
/// Thread-safe: the writer (mutated by add/delete/clear/commit) is guarded by a
/// [`Mutex`]; searches take no lock (tantivy searchers are `Send + Sync`).
pub struct TantivyFts {
    index: Index,
    writer: Mutex<IndexWriter>,
    reader: IndexReader,
    row_id_field: Field,
    content_field: Field,
    commit_fault: Mutex<Option<CommitFault>>,
    retry_delays: Mutex<Vec<Duration>>,
}

impl TantivyFts {
    /// In-memory index (tests, `:memory:` stores).
    pub fn in_memory() -> Result<Self, StoreError> {
        let schema = build_schema();
        let index = Index::create_in_ram(schema);
        Self::finish(index)
    }

    /// On-disk index rooted at `dir` (created if absent). If an existing index
    /// there is unreadable/version-incompatible, the directory is wiped and a
    /// fresh empty index is created. The returned flag is `true` in exactly that
    /// wipe-and-recreate case, so callers can repopulate from the source-of-truth
    /// tables via
    /// [`HistoryStore::rebuild_fts`](super::store::HistoryStore::rebuild_fts).
    pub fn open_dir(dir: &Path) -> Result<(Self, bool), StoreError> {
        std::fs::create_dir_all(dir).map_err(|e| StoreError::Fts(e.to_string()))?;
        let (index, recreated) = Self::open_or_recreate(dir)?;
        Ok((Self::finish(index)?, recreated))
    }

    /// Open the on-disk index, or wipe-and-recreate it if incompatible. The bool
    /// reports whether the recreate path was taken.
    fn open_or_recreate(dir: &Path) -> Result<(Index, bool), StoreError> {
        let schema = build_schema();
        let mmap = MmapDirectory::open(dir).map_err(|e| StoreError::Fts(e.to_string()))?;
        if let Ok(index) = Index::open_or_create(mmap, schema.clone()) {
            return Ok((index, false));
        }
        // Incompatible/corrupt on-disk index — wipe and start fresh.
        std::fs::remove_dir_all(dir).map_err(|e| StoreError::Fts(e.to_string()))?;
        std::fs::create_dir_all(dir).map_err(|e| StoreError::Fts(e.to_string()))?;
        let mmap = MmapDirectory::open(dir).map_err(|e| StoreError::Fts(e.to_string()))?;
        let index = Index::open_or_create(mmap, schema).map_err(|e| StoreError::Fts(e.to_string()))?;
        Ok((index, true))
    }

    fn finish(index: Index) -> Result<Self, StoreError> {
        index.tokenizers().register(ANALYZER_NAME, build_analyzer());
        let schema = index.schema();
        let row_id_field = schema
            .get_field("row_id")
            .map_err(|e| StoreError::Fts(e.to_string()))?;
        let content_field = schema
            .get_field("content")
            .map_err(|e| StoreError::Fts(e.to_string()))?;
        let writer = index
            .writer_with_num_threads::<TantivyDocument>(1, WRITER_HEAP_BYTES)
            .map_err(|e| StoreError::Fts(e.to_string()))?;
        // Manual reload so every search sees the immediately-preceding commit
        // (Python calls `index.reload()` after each write).
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e: tantivy::TantivyError| StoreError::Fts(e.to_string()))?;
        Ok(Self {
            index,
            writer: Mutex::new(writer),
            reader,
            row_id_field,
            content_field,
            commit_fault: Mutex::new(None),
            retry_delays: Mutex::new(COMMIT_RETRY_DELAYS.to_vec()),
        })
    }

    /// Whether the index currently holds no documents. Used after `open_dir` to
    /// decide whether a freshly-created (but not recreate-flagged) index needs
    /// repopulating from the source tables.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.reader.searcher().num_docs() == 0
    }

    /// Test seam: install (or clear with `None`) a fake commit that fails before
    /// the real commit runs. Mirrors monkeypatching `FtsIndex._commit_writer`.
    pub fn set_commit_fault(&self, fault: Option<CommitFault>) {
        *self
            .commit_fault
            .lock()
            .expect("commit_fault mutex poisoned") = fault;
    }

    /// Test seam: override the retry backoffs (e.g. to zero) so retry tests are
    /// fast. Empty means "no backoff iterations; one final attempt".
    pub fn set_retry_delays(&self, delays: Vec<Duration>) {
        *self
            .retry_delays
            .lock()
            .expect("retry_delays mutex poisoned") = delays;
    }

    fn document(&self, row_id: i64, content: &str) -> TantivyDocument {
        let mut doc = TantivyDocument::new();
        doc.add_i64(self.row_id_field, row_id);
        doc.add_text(self.content_field, content);
        doc
    }

    /// One commit attempt: consult the fault injector first, else the real
    /// writer commit.
    fn commit_once(&self, writer: &mut IndexWriter) -> Result<(), CommitErr> {
        if let Some(fault) = self
            .commit_fault
            .lock()
            .expect("commit_fault mutex poisoned")
            .as_mut()
        {
            if let Some(err) = fault() {
                return Err(CommitErr::Injected(err));
            }
        }
        writer.commit().map(|_| ()).map_err(CommitErr::Tantivy)
    }

    /// Commit with transient-lock retry (behavior 22).
    fn commit(&self, writer: &mut IndexWriter) -> Result<(), StoreError> {
        let delays = self
            .retry_delays
            .lock()
            .expect("retry_delays mutex poisoned")
            .clone();
        for delay in delays {
            match self.commit_once(writer) {
                Ok(()) => return Ok(()),
                Err(err) if err.is_transient_lock() => std::thread::sleep(delay),
                Err(err) => return Err(err.into_store_error()),
            }
        }
        // Final attempt — any error propagates.
        self.commit_once(writer)
            .map_err(CommitErr::into_store_error)
    }

    /// Commit (via the retry path) then reload the reader so reads are
    /// synchronous. `writer` is released before the reload (Python reloads
    /// outside the writer lock).
    fn commit_and_reload(
        &self,
        mut writer: std::sync::MutexGuard<'_, IndexWriter>,
    ) -> Result<(), StoreError> {
        self.commit(&mut writer)?;
        drop(writer);
        self.reader
            .reload()
            .map_err(|e| StoreError::Fts(e.to_string()))
    }
}

impl FtsIndex for TantivyFts {
    fn add(&self, row_id: i64, content: &str) -> Result<(), StoreError> {
        let writer = self.writer.lock().expect("fts writer mutex poisoned");
        writer.delete_term(Term::from_field_i64(self.row_id_field, row_id));
        writer
            .add_document(self.document(row_id, content))
            .map_err(|e| StoreError::Fts(e.to_string()))?;
        self.commit_and_reload(writer)
    }

    fn add_many(&self, rows: &[(i64, String)]) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }
        let writer = self.writer.lock().expect("fts writer mutex poisoned");
        for (row_id, content) in rows {
            writer.delete_term(Term::from_field_i64(self.row_id_field, *row_id));
            writer
                .add_document(self.document(*row_id, content))
                .map_err(|e| StoreError::Fts(e.to_string()))?;
        }
        self.commit_and_reload(writer)
    }

    fn delete(&self, row_id: i64) -> Result<(), StoreError> {
        let writer = self.writer.lock().expect("fts writer mutex poisoned");
        writer.delete_term(Term::from_field_i64(self.row_id_field, row_id));
        self.commit_and_reload(writer)
    }

    fn clear(&self) -> Result<(), StoreError> {
        let writer = self.writer.lock().expect("fts writer mutex poisoned");
        writer
            .delete_all_documents()
            .map_err(|e| StoreError::Fts(e.to_string()))?;
        self.commit_and_reload(writer)
    }

    fn search(&self, query: &str, limit: usize) -> Vec<(i64, f32)> {
        if limit == 0 || query.trim().is_empty() {
            return Vec::new();
        }
        let cleaned = sanitize_query(query);
        if cleaned.trim().is_empty() {
            return Vec::new();
        }
        let parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        // Residual parse errors → empty (never raise); disjunctive over content.
        let Ok(parsed) = parser.parse_query(&cleaned) else {
            return Vec::new();
        };
        let searcher = self.reader.searcher();
        let collector = TopDocs::with_limit(limit).order_by_score();
        let hits: Vec<(f32, tantivy::DocAddress)> = match searcher.search(&parsed, &collector) {
            Ok(hits) => hits,
            Err(_) => return Vec::new(),
        };
        let mut out: Vec<(i64, f32)> = Vec::with_capacity(hits.len());
        for (score, address) in hits {
            let Ok(doc) = searcher.doc::<TantivyDocument>(address) else {
                continue;
            };
            if let Some(row_id) = doc.get_first(self.row_id_field).and_then(|v| v.as_i64()) {
                out.push((row_id, score));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::{FTS_STOPWORDS, FtsIndex, TantivyFts, build_analyzer, sanitize_query};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn stopword_list_matches_python() {
        // Python's `_FTS_STOPWORDS` actually holds 87 entries (the spec's "88"
        // is off by one); parity requires the exact same set.
        assert_eq!(FTS_STOPWORDS.len(), 87);
    }

    #[test]
    fn analyzer_builds() {
        // Must not panic assembling the filter chain.
        let _ = build_analyzer();
    }

    #[test]
    fn sanitize_strips_punctuation_and_lowercases() {
        assert_eq!(sanitize_query("Fox's tail!"), "fox s tail ");
        assert_eq!(sanitize_query("FOX AND BEAR"), "fox and bear");
    }

    #[test]
    fn add_and_search_round_trip() {
        let idx = TantivyFts::in_memory().unwrap();
        idx.add(1, "the quick brown fox").unwrap();
        idx.add(2, "i like strawberry jam").unwrap();
        let hits = idx.search("fox", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 1);
        assert!(hits[0].1 > 0.0, "BM25 score must be positive");
    }

    #[test]
    fn open_dir_flags_recreate_on_incompatible_schema() {
        use tantivy::Index;
        use tantivy::schema::{STORED, STRING, Schema};

        let dir = tempfile::tempdir().unwrap();
        // Write an index whose schema differs from `build_schema` (row_id as text
        // rather than i64) — stands in for the Python-vs-Rust incompatibility.
        let mut sb = Schema::builder();
        sb.add_text_field("row_id", STRING | STORED);
        sb.add_text_field("content", STRING | STORED);
        Index::create_in_dir(dir.path(), sb.build()).unwrap();

        let (idx, recreated) = TantivyFts::open_dir(dir.path()).unwrap();
        assert!(recreated, "incompatible on-disk schema must trigger recreate");
        assert!(idx.is_empty(), "a recreated index starts empty");
    }

    #[test]
    fn open_dir_preserves_compatible_index() {
        let dir = tempfile::tempdir().unwrap();
        {
            let (idx, recreated) = TantivyFts::open_dir(dir.path()).unwrap();
            assert!(!recreated);
            idx.add(1, "the quick brown fox").unwrap();
        }
        let (idx, recreated) = TantivyFts::open_dir(dir.path()).unwrap();
        assert!(!recreated, "a compatible reopen must not wipe the index");
        assert!(!idx.is_empty());
        assert_eq!(idx.search("fox", 5).len(), 1);
    }

    #[test]
    fn upsert_replaces_prior_doc() {
        let idx = TantivyFts::in_memory().unwrap();
        idx.add(1, "the quick brown fox").unwrap();
        idx.add(1, "a rainy day in seattle").unwrap();
        assert!(idx.search("fox", 10).is_empty());
        assert_eq!(idx.search("seattle", 10).len(), 1);
    }

    #[test]
    fn stemming_recall_fox_foxes() {
        let idx = TantivyFts::in_memory().unwrap();
        idx.add(1, "anything with foxes is cool").unwrap();
        assert_eq!(idx.search("fox", 10).len(), 1);
    }

    #[test]
    fn empty_and_stopword_only_queries_return_nothing() {
        let idx = TantivyFts::in_memory().unwrap();
        idx.add(1, "the quick brown fox").unwrap();
        assert!(idx.search("", 10).is_empty());
        assert!(idx.search("   ", 10).is_empty());
        assert!(idx.search("hi the a do you", 10).is_empty());
        assert!(idx.search("fox", 0).is_empty());
    }

    #[test]
    fn retry_recovers_after_transient_lock() {
        let idx = TantivyFts::in_memory().unwrap();
        // fail on the first two attempts, succeed on the third (zero backoffs)
        idx.set_retry_delays(vec![std::time::Duration::ZERO; 3]);
        let calls = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&calls);
        idx.set_commit_fault(Some(Box::new(move || {
            let n = c.fetch_add(1, Ordering::SeqCst) + 1;
            if n < 3 {
                Some(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "Access is denied. os error 5 PermissionDenied",
                ))
            } else {
                None
            }
        })));
        idx.add(42, "the quick brown fox").unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert!(idx.search("fox", 5).iter().any(|(id, _)| *id == 42));
    }

    #[test]
    fn retry_exhausts_then_propagates() {
        let idx = TantivyFts::in_memory().unwrap();
        idx.set_retry_delays(vec![std::time::Duration::ZERO; 3]);
        let calls = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&calls);
        idx.set_commit_fault(Some(Box::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
            Some(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "Failed to open file for write: PermissionDenied",
            ))
        })));
        let err = idx.add(1, "hello").unwrap_err();
        assert!(format!("{err}").contains("PermissionDenied"));
        assert!(calls.load(Ordering::SeqCst) >= 3);
    }

    #[test]
    fn unrelated_commit_error_fails_fast() {
        let idx = TantivyFts::in_memory().unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&calls);
        idx.set_commit_fault(Some(Box::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
            Some(std::io::Error::other("schema mismatch"))
        })));
        let err = idx.add(1, "hello").unwrap_err();
        assert!(format!("{err}").contains("schema mismatch"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
