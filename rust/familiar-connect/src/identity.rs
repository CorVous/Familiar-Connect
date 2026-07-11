//! Author identity + canonical_key / slug / label + ego keys (subsystem 02;
//! Python `identity.py`).
//!
//! Replaces a bare `speaker: String` threading history, context, memory,
//! providers. [`Author`] carries an immutable platform key (`platform` +
//! `user_id`) plus human-readable variants (`username`, `display_name`,
//! `aliases`) — recall resolves by any known name; storage pins to the stable
//! [`Author::slug`]. See `docs/architecture/memory.md` for `people/<slug>.md`
//! usage.

use crate::llm::sanitize_name;
use regex::Regex;
use std::collections::BTreeSet;
use std::sync::LazyLock;

static SLUG_NON_ALNUM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[^a-z0-9]+").expect("valid slug regex"));

/// Reserved platform namespace for the familiar's OWN narrative subject.
///
/// `ego:` can never collide with `discord:` / `twitch:` keys — those use a real
/// platform name. One canonical key per familiar: `ego:<id>`. Named `ego` rather
/// than `self` so it never shadows a `self` parameter in surrounding code.
pub const EGO_PLATFORM: &str = "ego";

/// Reserved subject key for `familiar_id`'s own narrative: `ego:<id>`.
#[must_use]
pub fn ego_canonical_key(familiar_id: &str) -> String {
    format!("{EGO_PLATFORM}:{familiar_id}")
}

/// Test `ego:<id>` membership: the `ego` platform plus a non-empty id.
///
/// `"ego"` (no colon) is NOT an ego key; neither is `"ego:"` (empty id).
#[must_use]
pub fn is_ego_key(canonical_key: &str) -> bool {
    match canonical_key.split_once(':') {
        Some((platform, rest)) => platform == EGO_PLATFORM && !rest.is_empty(),
        None => false,
    }
}

/// Minimal Discord `Member`/`User` surface for [`Author::from_discord_member`].
///
/// `discord.Member` carries four name fields: `id` (immutable snowflake), `name`
/// (global username), `global_name` (global display name, 2023+), `nick`
/// (per-guild override; `None` on DMs). `display_name` is the resolved view
/// (`nick → global_name → name`). The optional accessors default to `None` so DM
/// `User` objects, older shells, and small test doubles satisfy the trait by
/// implementing only the three required methods.
pub trait DiscordMemberLike {
    /// Immutable snowflake id.
    fn id(&self) -> u64;
    /// Global username (`.name`).
    fn name(&self) -> String;
    /// Resolved display name (`nick → global_name → name`).
    fn display_name(&self) -> String;
    /// Global display name, when present.
    fn global_name(&self) -> Option<String> {
        None
    }
    /// Per-guild nickname, when present.
    fn nick(&self) -> Option<String> {
        None
    }
    /// Profile pronouns, when present.
    fn pronouns(&self) -> Option<String> {
        None
    }
    /// Profile bio, when present.
    fn bio(&self) -> Option<String> {
        None
    }
}

/// Platform-scoped identity for one speaker.
///
/// [`Author::canonical_key`] is the storage-stable id; [`Author::label`] the
/// human-readable string for prompts; `aliases` + the on-disk index map many
/// display names / nicknames onto one author. Immutable once constructed
/// (Python's frozen dataclass).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Author {
    /// `"discord"` / `"twitch"` / `"ego"` (kept as a string so slug
    /// normalization is exercised directly; see the tests).
    pub platform: String,
    /// Platform-scoped user id.
    pub user_id: String,
    /// Global username, when known.
    pub username: Option<String>,
    /// Resolved display name, when known.
    pub display_name: Option<String>,
    /// Global display name (Discord 2023+), when known.
    pub global_name: Option<String>,
    /// Per-guild nickname, when known.
    pub guild_nick: Option<String>,
    /// Profile pronouns, when known.
    pub pronouns: Option<String>,
    /// Profile bio, when known.
    pub bio: Option<String>,
    /// Every alias mapped onto this author.
    pub aliases: BTreeSet<String>,
}

impl Author {
    /// Construct an author with the four core fields; the rest default to
    /// `None` / empty.
    #[must_use]
    pub fn new(
        platform: impl Into<String>,
        user_id: impl Into<String>,
        username: Option<String>,
        display_name: Option<String>,
    ) -> Self {
        Self {
            platform: platform.into(),
            user_id: user_id.into(),
            username,
            display_name,
            global_name: None,
            guild_nick: None,
            pronouns: None,
            bio: None,
            aliases: BTreeSet::new(),
        }
    }

    /// Builder: replace the alias set.
    #[must_use]
    pub fn with_aliases(mut self, aliases: impl IntoIterator<Item = String>) -> Self {
        self.aliases = aliases.into_iter().collect();
        self
    }

    /// Stable identifier across renames: `<platform>:<user_id>`.
    #[must_use]
    pub fn canonical_key(&self) -> String {
        format!("{}:{}", self.platform, self.user_id)
    }

    /// Filesystem-safe form of [`Author::canonical_key`]: lowercase, runs of
    /// non-alphanumerics collapsed to single dashes, ends trimmed. Basename for
    /// `people/<slug>.md`.
    #[must_use]
    pub fn slug(&self) -> String {
        let lowered = self.canonical_key().to_lowercase();
        let dashed = SLUG_NON_ALNUM.replace_all(&lowered, "-");
        dashed.trim_matches('-').to_owned()
    }

    /// Preferred display string: `display_name → username → user_id` (first
    /// non-empty).
    #[must_use]
    pub fn label(&self) -> String {
        if let Some(d) = self.display_name.as_deref() {
            if !d.is_empty() {
                return d.to_owned();
            }
        }
        if let Some(u) = self.username.as_deref() {
            if !u.is_empty() {
                return u.to_owned();
            }
        }
        self.user_id.clone()
    }

    /// OpenAI-style `Message.name`, scrubbed per API rules. Falls back to a
    /// sanitized `user_id` when the label scrubs empty (e.g. an all-punctuation
    /// display name). `None` only if the id also scrubs empty — not expected.
    #[must_use]
    pub fn openai_name(&self) -> Option<String> {
        sanitize_name(&self.label()).or_else(|| sanitize_name(&self.user_id))
    }

    /// Every known name (aliases ∪ display_name ∪ username), used to rebuild the
    /// alias index. Empty names are dropped.
    #[must_use]
    pub fn all_known_names(&self) -> BTreeSet<String> {
        let mut names = self.aliases.clone();
        if let Some(d) = self.display_name.as_deref() {
            if !d.is_empty() {
                names.insert(d.to_owned());
            }
        }
        if let Some(u) = self.username.as_deref() {
            if !u.is_empty() {
                names.insert(u.to_owned());
            }
        }
        names
    }

    /// Build from a Discord `Member` / `User`. Optional fields
    /// (`global_name` / `nick` / `pronouns` / `bio`) read defensively — absent
    /// ones stay `None`.
    pub fn from_discord_member(member: &impl DiscordMemberLike) -> Self {
        Self {
            platform: "discord".to_owned(),
            user_id: member.id().to_string(),
            username: Some(member.name()),
            display_name: Some(member.display_name()),
            global_name: member.global_name(),
            guild_nick: member.nick(),
            pronouns: member.pronouns(),
            bio: member.bio(),
            aliases: BTreeSet::new(),
        }
    }

    /// Build from Twitch Helix fields. `user_login` is the lowercase immutable
    /// login (→ `username`); `user_name` is the mutable display case
    /// (→ `display_name`).
    #[must_use]
    pub fn from_twitch(
        user_id: impl Into<String>,
        user_login: Option<String>,
        user_name: Option<String>,
    ) -> Self {
        Self {
            platform: "twitch".to_owned(),
            user_id: user_id.into(),
            username: user_login,
            display_name: user_name,
            global_name: None,
            guild_nick: None,
            pronouns: None,
            bio: None,
            aliases: BTreeSet::new(),
        }
    }
}

/// Render one turn as `role (label): content` (user turns with an author) or
/// `role: content` (everything else).
///
/// Shared between the history-summary provider (05) and the memory writer (07);
/// the format is a cross-module contract — changing it invalidates stored
/// summaries' framing.
#[must_use]
pub fn format_turn_for_transcript(role: &str, author: Option<&Author>, content: &str) -> String {
    if role == "user" {
        if let Some(a) = author {
            return format!("{role} ({}): {content}", a.label());
        }
    }
    format!("{role}: {content}")
}

#[cfg(test)]
mod tests {
    use super::{
        Author, DiscordMemberLike, EGO_PLATFORM, ego_canonical_key, format_turn_for_transcript,
        is_ego_key,
    };
    use std::collections::BTreeSet;

    fn names(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    /// A full Discord member double: id + name + display_name + optionals.
    struct FullMember {
        id: u64,
        name: String,
        display_name: String,
        global_name: Option<String>,
        nick: Option<String>,
        pronouns: Option<String>,
        bio: Option<String>,
    }

    impl DiscordMemberLike for FullMember {
        fn id(&self) -> u64 {
            self.id
        }
        fn name(&self) -> String {
            self.name.clone()
        }
        fn display_name(&self) -> String {
            self.display_name.clone()
        }
        fn global_name(&self) -> Option<String> {
            self.global_name.clone()
        }
        fn nick(&self) -> Option<String> {
            self.nick.clone()
        }
        fn pronouns(&self) -> Option<String> {
            self.pronouns.clone()
        }
        fn bio(&self) -> Option<String> {
            self.bio.clone()
        }
    }

    /// A minimal member double (id + name + display_name only) — exercises the
    /// default `None`-returning trait accessors (older py-cord shapes / DM Users).
    struct MinimalMember {
        id: u64,
        name: String,
        display_name: String,
    }

    impl DiscordMemberLike for MinimalMember {
        fn id(&self) -> u64 {
            self.id
        }
        fn name(&self) -> String {
            self.name.clone()
        }
        fn display_name(&self) -> String {
            self.display_name.clone()
        }
    }

    // --- construction ------------------------------------------------------

    #[test]
    fn required_fields() {
        let a = Author::new("discord", "123", Some("ada".into()), Some("Ada".into()));
        assert_eq!(a.platform, "discord");
        assert_eq!(a.user_id, "123");
        assert_eq!(a.username.as_deref(), Some("ada"));
        assert_eq!(a.display_name.as_deref(), Some("Ada"));
        assert!(a.aliases.is_empty());
    }

    #[test]
    fn aliases_default_empty() {
        let a = Author::new("discord", "1", None, None);
        assert!(a.aliases.is_empty());
    }

    #[test]
    fn aliases_accept_iterable() {
        let a = Author::new("discord", "1", Some("ada".into()), Some("Ada".into()))
            .with_aliases(["Addy".to_owned(), "A".to_owned()]);
        assert!(a.aliases.contains("Addy"));
        assert!(a.aliases.contains("A"));
    }

    // --- canonical_key + slug ---------------------------------------------

    #[test]
    fn canonical_key_is_platform_colon_id() {
        let a = Author::new("discord", "42", None, None);
        assert_eq!(a.canonical_key(), "discord:42");
    }

    #[test]
    fn slug_replaces_colon_and_lowercases() {
        // Arbitrary platform casing exercises slug normalization, not a real key.
        let a = Author::new("Discord", "42", None, None);
        assert_eq!(a.slug(), "discord-42");
    }

    #[test]
    fn slug_strips_surrounding_dashes() {
        let a = Author::new(":x:", ":7:", None, None);
        assert_eq!(a.slug(), "x-7");
    }

    #[test]
    fn slug_collapses_runs_of_non_alphanumerics() {
        let a = Author::new("twitch", "U__99!!", None, None);
        assert_eq!(a.slug(), "twitch-u-99");
    }

    // --- label -------------------------------------------------------------

    #[test]
    fn label_prefers_display_name() {
        let a = Author::new(
            "discord",
            "1",
            Some("ada_l".into()),
            Some("Ada Lovelace".into()),
        );
        assert_eq!(a.label(), "Ada Lovelace");
    }

    #[test]
    fn label_falls_back_to_username() {
        let a = Author::new("discord", "1", Some("ada_l".into()), None);
        assert_eq!(a.label(), "ada_l");
    }

    #[test]
    fn label_falls_back_to_user_id() {
        let a = Author::new("discord", "1", None, None);
        assert_eq!(a.label(), "1");
    }

    // --- openai_name -------------------------------------------------------

    #[test]
    fn sanitizes_display_name() {
        let a = Author::new("discord", "1", None, Some("Ada Lovelace!".into()));
        assert_eq!(a.openai_name().as_deref(), Some("Ada_Lovelace"));
    }

    #[test]
    fn falls_back_to_user_id_when_label_sanitizes_to_empty() {
        // An all-fullwidth-punctuation display name scrubs to empty.
        let a = Author::new("discord", "42", None, Some("！！！".into()));
        assert_eq!(a.openai_name().as_deref(), Some("42"));
    }

    // --- all_known_names ---------------------------------------------------

    #[test]
    fn all_known_names_includes_display_username_and_aliases() {
        let a = Author::new("discord", "1", Some("ada_l".into()), Some("Ada".into()))
            .with_aliases(["Addy".to_owned()]);
        assert_eq!(a.all_known_names(), names(&["Ada", "ada_l", "Addy"]));
    }

    #[test]
    fn all_known_names_drops_none_values() {
        let a = Author::new("discord", "1", None, Some("Ada".into()));
        assert_eq!(a.all_known_names(), names(&["Ada"]));
    }

    // --- from_discord_member ----------------------------------------------

    #[test]
    fn extracts_id_name_display_name() {
        let member = MinimalMember {
            id: 987,
            name: "ada_l".into(),
            display_name: "Ada".into(),
        };
        let a = Author::from_discord_member(&member);
        assert_eq!(a.platform, "discord");
        assert_eq!(a.user_id, "987");
        assert_eq!(a.username.as_deref(), Some("ada_l"));
        assert_eq!(a.display_name.as_deref(), Some("Ada"));
    }

    #[test]
    fn extracts_global_name_and_guild_nick() {
        let member = FullMember {
            id: 987,
            name: "ada_l".into(),
            display_name: "Aria".into(),
            global_name: Some("Ada Lovelace".into()),
            nick: Some("Aria".into()),
            pronouns: None,
            bio: None,
        };
        let a = Author::from_discord_member(&member);
        assert_eq!(a.global_name.as_deref(), Some("Ada Lovelace"));
        assert_eq!(a.guild_nick.as_deref(), Some("Aria"));
        assert_eq!(a.display_name.as_deref(), Some("Aria"));
    }

    #[test]
    fn handles_member_without_global_name_or_nick() {
        let member = MinimalMember {
            id: 987,
            name: "ada_l".into(),
            display_name: "ada_l".into(),
        };
        let a = Author::from_discord_member(&member);
        assert!(a.global_name.is_none());
        assert!(a.guild_nick.is_none());
    }

    #[test]
    fn extracts_pronouns_and_bio_when_present() {
        let member = FullMember {
            id: 987,
            name: "ada_l".into(),
            display_name: "Ada".into(),
            global_name: None,
            nick: None,
            pronouns: Some("she/her".into()),
            bio: Some("Designs analytical engines.".into()),
        };
        let a = Author::from_discord_member(&member);
        assert_eq!(a.pronouns.as_deref(), Some("she/her"));
        assert_eq!(a.bio.as_deref(), Some("Designs analytical engines."));
    }

    #[test]
    fn pronouns_and_bio_default_none() {
        let member = MinimalMember {
            id: 987,
            name: "ada_l".into(),
            display_name: "Ada".into(),
        };
        let a = Author::from_discord_member(&member);
        assert!(a.pronouns.is_none());
        assert!(a.bio.is_none());
    }

    // --- ego keys ----------------------------------------------------------

    #[test]
    fn builds_ego_prefixed_key() {
        assert_eq!(ego_canonical_key("sapphire"), "ego:sapphire");
        assert!(ego_canonical_key("sapphire").starts_with(&format!("{EGO_PLATFORM}:")));
    }

    #[test]
    fn round_trips_membership() {
        let key = ego_canonical_key("fam");
        assert!(is_ego_key(&key));
    }

    #[test]
    fn discord_keys_are_not_ego() {
        assert!(!is_ego_key("discord:123"));
        assert!(!is_ego_key("twitch:abc"));
        assert!(!is_ego_key(""));
        assert!(!is_ego_key("ego")); // bare prefix, no id portion
        assert!(!is_ego_key("ego:")); // colon but empty id
    }

    #[test]
    fn ego_key_does_not_collide_with_discord() {
        assert_ne!(ego_canonical_key("123"), "discord:123");
    }

    // --- from_twitch -------------------------------------------------------

    #[test]
    fn builds_from_twitch_api_fields() {
        let a = Author::from_twitch("U99", Some("adadev".into()), Some("AdaDev".into()));
        assert_eq!(a.platform, "twitch");
        assert_eq!(a.user_id, "U99");
        assert_eq!(a.username.as_deref(), Some("adadev"));
        assert_eq!(a.display_name.as_deref(), Some("AdaDev"));
    }

    // --- transcript formatting --------------------------------------------

    #[test]
    fn transcript_user_turn_includes_label() {
        let a = Author::new("discord", "1", Some("ada_l".into()), Some("Ada".into()));
        assert_eq!(
            format_turn_for_transcript("user", Some(&a), "hi"),
            "user (Ada): hi"
        );
    }

    #[test]
    fn transcript_non_user_turn_is_role_only() {
        let a = Author::new("discord", "1", None, Some("Ada".into()));
        assert_eq!(
            format_turn_for_transcript("assistant", Some(&a), "hi"),
            "assistant: hi"
        );
        assert_eq!(format_turn_for_transcript("user", None, "hi"), "user: hi");
    }
}
