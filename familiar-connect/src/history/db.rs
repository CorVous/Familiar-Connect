//! Dedicated single-owner DB actor over `rusqlite` (subsystem 03; supersedes
//! Python `history/turso_compat.py`).
//!
//! One OS thread owns the [`rusqlite::Connection`]. Every operation is a
//! whole-operation closure submitted over an `mpsc` channel; the caller blocks
//! on a per-call reply channel (DESIGN §4.4, decision D5). This preserves the
//! Python single-owning-thread contract — all SQL executes on one thread, never
//! the caller's — while letting multi-statement operations run in explicit
//! transactions (`&Connection` closures via [`Db::run`]).
//!
//! A statement-trace hook ([`Db::set_trace_callback`]) fires on the actor thread
//! just before each traced statement, mirroring Python's
//! `TursoConnection.set_trace_callback` (used by the single-query-count tests).
//!
//! Calls after [`Db::close`] fail with [`StoreError::Closed`] (behavior 7).

use std::path::Path;
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use rusqlite::{Connection, Row};

use super::StoreError;

/// Owned SQL bind value. Re-exported so tests can build raw-query params without
/// depending on `rusqlite` directly.
pub use rusqlite::types::Value;

/// Statement-trace hook: invoked with the SQL text on the actor thread just
/// before a traced statement runs (mirrors `sqlite3` / pyturso trace callbacks).
pub type TraceCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// A whole-operation closure handed to the actor thread.
type Job = Box<dyn FnOnce(&mut ActorState) + Send>;

/// State owned exclusively by the actor thread.
struct ActorState {
    conn: Connection,
    trace: Option<TraceCallback>,
}

/// Handle to the DB actor. Cloneable-by-`Arc` at the call sites that need it;
/// `Send + Sync` so the async facade can share it across worker tasks.
pub struct Db {
    sender: Mutex<Option<Sender<Job>>>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl Db {
    /// Open an in-memory database (`:memory:`).
    pub fn open_memory() -> Result<Self, StoreError> {
        Self::spawn(Connection::open_in_memory()?)
    }

    /// Open (creating if absent) a file-backed database at `path`.
    pub fn open(path: &Path) -> Result<Self, StoreError> {
        Self::spawn(Connection::open(path)?)
    }

    fn spawn(conn: Connection) -> Result<Self, StoreError> {
        let (tx, rx) = mpsc::channel::<Job>();
        let handle = std::thread::Builder::new()
            .name("history-db".to_owned())
            .spawn(move || {
                let mut state = ActorState { conn, trace: None };
                while let Ok(job) = rx.recv() {
                    job(&mut state);
                }
            })?;
        Ok(Self {
            sender: Mutex::new(Some(tx)),
            handle: Mutex::new(Some(handle)),
        })
    }

    /// Submit a job to the actor and block for its result.
    ///
    /// Returns [`StoreError::Closed`] if the actor has been shut down.
    fn dispatch<R, F>(&self, f: F) -> Result<R, StoreError>
    where
        F: FnOnce(&mut ActorState) -> R + Send + 'static,
        R: Send + 'static,
    {
        let (reply_tx, reply_rx) = mpsc::channel::<R>();
        let job: Job = Box::new(move |state| {
            let _ = reply_tx.send(f(state));
        });
        // Clone the sender out and release the lock before blocking on the
        // reply, so concurrent submissions don't serialize on the mutex.
        let sender = {
            let guard = self.sender.lock().unwrap();
            match guard.as_ref() {
                Some(sender) => sender.clone(),
                None => return Err(StoreError::Closed),
            }
        };
        sender.send(job).map_err(|_| StoreError::Closed)?;
        reply_rx.recv().map_err(|_| StoreError::Closed)
    }

    /// Run a whole-operation closure with direct `&Connection` access.
    ///
    /// Use for multi-statement atomic operations: open an explicit transaction
    /// via [`Connection::unchecked_transaction`] inside `f`. Not statement-traced
    /// (no test counts statements inside these operations).
    pub fn run<T, F>(&self, f: F) -> Result<T, StoreError>
    where
        F: FnOnce(&Connection) -> Result<T, StoreError> + Send + 'static,
        T: Send + 'static,
    {
        self.dispatch(move |state| f(&state.conn))?
    }

    /// Execute a single statement (auto-committed) and return the affected count.
    pub fn execute(&self, sql: impl Into<String>, params: Vec<Value>) -> Result<usize, StoreError> {
        let sql = sql.into();
        self.dispatch(move |state| {
            if let Some(cb) = &state.trace {
                cb(&sql);
            }
            state
                .conn
                .execute(&sql, rusqlite::params_from_iter(params.iter()))
                .map_err(StoreError::from)
        })?
    }

    /// Execute a single INSERT and return `last_insert_rowid` atomically.
    pub fn execute_returning_id(
        &self,
        sql: impl Into<String>,
        params: Vec<Value>,
    ) -> Result<i64, StoreError> {
        let sql = sql.into();
        self.dispatch(move |state| {
            if let Some(cb) = &state.trace {
                cb(&sql);
            }
            state
                .conn
                .execute(&sql, rusqlite::params_from_iter(params.iter()))?;
            Ok(state.conn.last_insert_rowid())
        })?
    }

    /// Run a query and map every row, collecting into a `Vec`.
    pub fn query_map<T, F>(
        &self,
        sql: impl Into<String>,
        params: Vec<Value>,
        mut mapf: F,
    ) -> Result<Vec<T>, StoreError>
    where
        F: FnMut(&Row) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let sql = sql.into();
        self.dispatch(move |state| {
            if let Some(cb) = &state.trace {
                cb(&sql);
            }
            let mut stmt = state.conn.prepare(&sql)?;
            let mapped = stmt.query_map(rusqlite::params_from_iter(params.iter()), |r| mapf(r))?;
            mapped
                .collect::<rusqlite::Result<Vec<T>>>()
                .map_err(StoreError::from)
        })?
    }

    /// Execute a multi-statement SQL script (schema setup / migrations).
    pub fn execute_batch(&self, sql: impl Into<String>) -> Result<(), StoreError> {
        let sql = sql.into();
        self.dispatch(move |state| state.conn.execute_batch(&sql).map_err(StoreError::from))?
    }

    /// Read a single optional `i64` scalar from column 0 (test/diagnostic helper).
    pub fn query_scalar_i64(
        &self,
        sql: impl Into<String>,
        params: Vec<Value>,
    ) -> Result<Option<i64>, StoreError> {
        Ok(self
            .query_map(sql, params, |r| r.get::<_, Option<i64>>(0))?
            .into_iter()
            .next()
            .flatten())
    }

    /// Read a single optional text scalar from column 0 (test/diagnostic helper).
    pub fn query_scalar_string(
        &self,
        sql: impl Into<String>,
        params: Vec<Value>,
    ) -> Result<Option<String>, StoreError> {
        Ok(self
            .query_map(sql, params, |r| r.get::<_, Option<String>>(0))?
            .into_iter()
            .next()
            .flatten())
    }

    /// Install (or clear with `None`) the statement-trace hook. The callback
    /// fires on the actor thread, before each traced statement.
    pub fn set_trace_callback(&self, cb: Option<TraceCallback>) {
        let _ = self.dispatch(move |state| {
            state.trace = cb;
        });
    }

    /// Shut down the actor thread. Idempotent; subsequent calls error with
    /// [`StoreError::Closed`].
    pub fn close(&self) {
        let sender = self.sender.lock().unwrap().take();
        drop(sender);
        let handle = self.handle.lock().unwrap().take();
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }
}

impl Drop for Db {
    fn drop(&mut self) {
        self.close();
    }
}
