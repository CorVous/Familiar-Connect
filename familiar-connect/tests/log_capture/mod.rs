//! Shared `tracing` capture harness for non-responder integration tests.
//!
//! Included via `#[path = "log_capture/mod.rs"] mod log_capture;`. Mirrors the
//! copy embedded in `responders_support/mod.rs` (the responder binaries pull it
//! from there); the two support modules are never linked into the same binary,
//! so the small duplication matches the repo's parallel-helper idiom.
//!
//! One process-global subscriber writes through a thread-local buffer slot.
//! Per-test `set_default` subscribers are a flake factory: every install/drop
//! rebuilds tracing's GLOBAL callsite-interest cache, and a rebuild landing
//! between another test's install and its emit — computed at an instant with no
//! live capture — caches the callsite as never-interested, silently dropping the
//! emit. A single global dispatcher computes interest exactly once; per-test
//! isolation comes from the thread-local sink instead (tests run one-per-thread;
//! the current-thread tokio runtime keeps emits on the test's own thread).

use std::sync::{Arc, Mutex};

std::thread_local! {
    static CAPTURE_SINK: std::cell::RefCell<Option<Arc<Mutex<Vec<u8>>>>> =
        const { std::cell::RefCell::new(None) };
}

#[derive(Clone, Copy)]
struct ThreadLocalWriter;

impl std::io::Write for ThreadLocalWriter {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        CAPTURE_SINK.with(|slot| {
            if let Some(buf) = slot.borrow().as_ref() {
                buf.lock().expect("log buf").extend_from_slice(data);
            }
        });
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for ThreadLocalWriter {
    type Writer = Self;
    fn make_writer(&'a self) -> Self::Writer {
        *self
    }
}

/// A per-test tracing capture; hold it for the duration of the assertion.
pub struct LogCapture {
    buf: Arc<Mutex<Vec<u8>>>,
}

impl LogCapture {
    /// Register this test's capture buffer (global subscriber installed once).
    pub fn install() -> Self {
        static GLOBAL: std::sync::Once = std::sync::Once::new();
        GLOBAL.call_once(|| {
            let subscriber = tracing_subscriber::fmt()
                .with_writer(ThreadLocalWriter)
                .with_ansi(true)
                .with_target(false)
                .with_max_level(tracing::Level::TRACE)
                .finish();
            tracing::subscriber::set_global_default(subscriber)
                .expect("no other global subscriber in the test binary");
        });
        let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
        CAPTURE_SINK.with(|slot| *slot.borrow_mut() = Some(buf.clone()));
        Self { buf }
    }
    /// The captured text so far.
    pub fn contents(&self) -> String {
        String::from_utf8_lossy(&self.buf.lock().expect("log buf")).into_owned()
    }
}

impl Drop for LogCapture {
    fn drop(&mut self) {
        CAPTURE_SINK.with(|slot| *slot.borrow_mut() = None);
    }
}
