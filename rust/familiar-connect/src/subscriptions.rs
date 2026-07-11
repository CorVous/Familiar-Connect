//! Persistent subscription registry backed by a TOML sidecar (subsystem 02;
//! Python `subscriptions.py`).
//!
//! Multi-channel, multi-kind. Every persisting mutation rewrites the whole file
//! (tens of rows at most) so subscriptions survive restart, and the file stays
//! human-editable on disk. Load is tolerant of content problems (a hand-edit
//! must not brick startup) but strict on TOML syntax (a parse error propagates)
//! — preserve that asymmetry.
//!
//! The on-disk layout is byte-stable: a header comment, then one
//! `[[subscription]]` block per persisted row in `(channel_id, kind)` order,
//! with the `guild_id` line omitted when `None`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Errors from loading or persisting the sidecar. Content problems degrade to an
/// empty/ignored row and never surface here; only TOML syntax and I/O faults do.
#[derive(Debug, thiserror::Error)]
pub enum SubscriptionError {
    /// The sidecar file could not be read.
    #[error("failed to read subscription registry at {path}: {source}")]
    Read {
        /// Sidecar path.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The sidecar contained invalid TOML syntax.
    #[error("failed to parse subscription registry at {path}: {source}")]
    Parse {
        /// Sidecar path.
        path: String,
        /// Underlying TOML error.
        #[source]
        source: toml::de::Error,
    },
    /// The sidecar could not be written.
    #[error("failed to write subscription registry at {path}: {source}")]
    Write {
        /// Sidecar path.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Text or voice — distinct rows even when a channel hosts both.
///
/// Ordered `Text < Voice`, matching the `"text" < "voice"` string ordering the
/// on-disk sort uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SubscriptionKind {
    /// A text-channel subscription.
    Text,
    /// A voice-channel subscription.
    Voice,
}

impl SubscriptionKind {
    /// The wire string (`"text"` / `"voice"`).
    #[must_use]
    pub const fn value(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Voice => "voice",
        }
    }

    /// Parse a wire string, `None` for unknown kinds.
    #[must_use]
    pub fn from_value(s: &str) -> Option<Self> {
        match s {
            "text" => Some(Self::Text),
            "voice" => Some(Self::Voice),
            _ => None,
        }
    }
}

/// A single persistent subscription row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Subscription {
    /// Discord channel snowflake.
    pub channel_id: u64,
    /// Text or voice.
    pub kind: SubscriptionKind,
    /// Owning guild snowflake, when known.
    pub guild_id: Option<u64>,
}

/// In-memory set backed by a TOML sidecar.
///
/// Loads on construction; persisting mutations rewrite the whole file. Rows are
/// keyed by `(channel_id, kind)`, so text and voice for the same channel
/// coexist. Ephemeral rows (`persist=false`) are queryable but never written.
#[derive(Debug)]
pub struct SubscriptionRegistry {
    path: PathBuf,
    rows: BTreeMap<(u64, SubscriptionKind), Subscription>,
    ephemeral: BTreeSet<(u64, SubscriptionKind)>,
}

impl SubscriptionRegistry {
    /// Construct a registry, loading the sidecar at `path`.
    ///
    /// A missing file yields an empty registry. Content problems (non-table
    /// rows, wrong-typed fields, unknown kinds) are tolerated per-row; a TOML
    /// syntax error is returned as [`SubscriptionError::Parse`].
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, SubscriptionError> {
        let mut reg = Self {
            path: path.into(),
            rows: BTreeMap::new(),
            ephemeral: BTreeSet::new(),
        };
        reg.load()?;
        Ok(reg)
    }

    // -- Queries -----------------------------------------------------------

    /// Snapshot of every registered subscription (persisted and ephemeral).
    #[must_use]
    pub fn all(&self) -> Vec<Subscription> {
        self.rows.values().copied().collect()
    }

    /// The subscription for `(channel_id, kind)`, if any.
    #[must_use]
    pub fn get(&self, channel_id: u64, kind: SubscriptionKind) -> Option<Subscription> {
        self.rows.get(&(channel_id, kind)).copied()
    }

    /// The first kind subscribed for `channel_id` — text before voice — or
    /// `None`.
    #[must_use]
    pub fn kind_for(&self, channel_id: u64) -> Option<SubscriptionKind> {
        // Declaration order (Text before Voice) is the tie-break.
        [SubscriptionKind::Text, SubscriptionKind::Voice]
            .into_iter()
            .find(|&kind| self.rows.contains_key(&(channel_id, kind)))
    }

    /// The voice subscription in `guild_id`, if any (at most one by convention).
    #[must_use]
    pub fn voice_in_guild(&self, guild_id: u64) -> Option<Subscription> {
        self.rows
            .values()
            .find(|sub| sub.kind == SubscriptionKind::Voice && sub.guild_id == Some(guild_id))
            .copied()
    }

    // -- Mutations (each persisting change rewrites the whole file) --------

    /// Add or replace `(channel_id, kind)`; idempotent upsert.
    ///
    /// Re-add updates `guild_id`. With `persist = false` the row is registered in
    /// memory only and never written — even when a later persisted mutation
    /// rewrites the file. A subsequent `persist = true` add of the same key
    /// promotes it to persisted.
    pub fn add(
        &mut self,
        channel_id: u64,
        kind: SubscriptionKind,
        guild_id: Option<u64>,
        persist: bool,
    ) -> Result<Subscription, SubscriptionError> {
        let key = (channel_id, kind);
        let sub = Subscription {
            channel_id,
            kind,
            guild_id,
        };
        self.rows.insert(key, sub);
        if persist {
            self.ephemeral.remove(&key);
            self.save()?;
        } else {
            self.ephemeral.insert(key);
        }
        Ok(sub)
    }

    /// Remove `(channel_id, kind)`; no-op if absent. Only writes when a row was
    /// actually removed.
    pub fn remove(
        &mut self,
        channel_id: u64,
        kind: SubscriptionKind,
    ) -> Result<(), SubscriptionError> {
        let key = (channel_id, kind);
        if self.rows.remove(&key).is_some() {
            self.ephemeral.remove(&key);
            self.save()?;
        }
        Ok(())
    }

    // -- Persistence -------------------------------------------------------

    fn load(&mut self) -> Result<(), SubscriptionError> {
        if !self.path.exists() {
            return Ok(());
        }
        let text =
            std::fs::read_to_string(&self.path).map_err(|source| SubscriptionError::Read {
                path: self.path.display().to_string(),
                source,
            })?;
        let value: toml::Value =
            toml::from_str(&text).map_err(|source| SubscriptionError::Parse {
                path: self.path.display().to_string(),
                source,
            })?;
        let Some(rows) = value.get("subscription").and_then(toml::Value::as_array) else {
            return Ok(());
        };
        for row in rows {
            let Some(table) = row.as_table() else {
                continue; // non-table rows skipped
            };
            let Some(channel_id) = table
                .get("channel_id")
                .and_then(toml::Value::as_integer)
                .and_then(|n| u64::try_from(n).ok())
            else {
                continue; // missing / wrong-typed channel_id
            };
            let Some(kind) = table
                .get("kind")
                .and_then(toml::Value::as_str)
                .and_then(SubscriptionKind::from_value)
            else {
                continue; // missing / wrong-typed / unknown kind
            };
            let guild_id = table
                .get("guild_id")
                .and_then(toml::Value::as_integer)
                .and_then(|n| u64::try_from(n).ok());
            self.rows.insert(
                (channel_id, kind),
                Subscription {
                    channel_id,
                    kind,
                    guild_id,
                },
            );
        }
        Ok(())
    }

    fn save(&self) -> Result<(), SubscriptionError> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|source| SubscriptionError::Write {
                    path: self.path.display().to_string(),
                    source,
                })?;
            }
        }
        std::fs::write(&self.path, self.serialize()).map_err(|source| SubscriptionError::Write {
            path: self.path.display().to_string(),
            source,
        })
    }

    /// Render the byte-exact sidecar contents (header + sorted `[[subscription]]`
    /// blocks, ephemeral rows excluded).
    fn serialize(&self) -> String {
        let header = "# Persistent subscription registry.\n\
             # Managed by /subscribe-* slash commands; safe to hand-edit while the bot is stopped.\n";
        let mut lines: Vec<String> = vec![header.to_owned()];
        // BTreeMap iteration is already sorted by (channel_id, kind).
        for (key, sub) in &self.rows {
            if self.ephemeral.contains(key) {
                continue;
            }
            let mut row_lines = vec![
                "[[subscription]]".to_owned(),
                format!("channel_id = {}", sub.channel_id),
                format!("kind = \"{}\"", sub.kind.value()),
            ];
            if let Some(guild_id) = sub.guild_id {
                row_lines.push(format!("guild_id = {guild_id}"));
            }
            row_lines.push(String::new());
            lines.push(row_lines.join("\n"));
        }
        lines.join("\n")
    }
}

impl AsRef<Path> for SubscriptionRegistry {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}
