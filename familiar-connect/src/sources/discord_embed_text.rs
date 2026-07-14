//! Render Discord embeds as plain text (subsystem 10; Python
//! `sources/discord_embed_text.py`).
//!
//! Discord delivers URL unfurls as `Embed` objects on a message. A bot only sees
//! `message.content` by default — link previews vanish. [`format_embeds`] flattens
//! a slice of embeds into a text block the shell appends to `content` before
//! publishing onto the bus, so the LLM sees the same body humans see in the
//! client.
//!
//! Python duck-types the input (any object exposing the relevant attributes). The
//! Rust port keeps the same freedom by taking a small structural [`EmbedView`]
//! input struct (DESIGN port notes: "duck-typing becomes small input structs …
//! tests then build them literally"); the serenity glue constructs one from a
//! `serenity::model::channel::Embed`.

/// The literal marker prefixing every rendered embed block.
const EMBED_TAG: &str = "[embed]";

/// One embed field (`name` / `value`), each optionally absent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmbedFieldView {
    /// Field name.
    pub name: Option<String>,
    /// Field value.
    pub value: Option<String>,
}

/// An embed's image reference. `proxy_url` (Discord's re-hosted copy) is preferred
/// over `url` when collecting images (see `bot::collect_images`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmbedImageView {
    /// Original source URL.
    pub url: Option<String>,
    /// Discord-proxied URL (more reliably fetchable).
    pub proxy_url: Option<String>,
}

/// Structural, duck-typed view of a Discord embed.
///
/// Every field is optional so tests build only the ones a case needs, exactly as
/// the Python doubles do with `SimpleNamespace`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EmbedView {
    /// `embed.provider.name` (unfurl source, e.g. `Tumblr`).
    pub provider_name: Option<String>,
    /// `embed.author.name`.
    pub author_name: Option<String>,
    /// `embed.title`.
    pub title: Option<String>,
    /// `embed.description`.
    pub description: Option<String>,
    /// `embed.footer.text`.
    pub footer_text: Option<String>,
    /// `embed.url` (image-only fallback link target).
    pub url: Option<String>,
    /// `embed.fields`.
    pub fields: Vec<EmbedFieldView>,
    /// `embed.image` (read only by image collection).
    pub image: Option<EmbedImageView>,
}

/// Flatten `embeds` into a plain-text block.
///
/// Empty input or all-blank embeds → `""`. Multiple embeds are separated by a
/// blank line so the LLM can tell them apart.
#[must_use]
pub fn format_embeds(embeds: &[EmbedView]) -> String {
    embeds
        .iter()
        .map(format_one)
        .filter(|b| !b.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Coerce an optional string to its stripped form, or `""`.
fn stripped(value: Option<&String>) -> String {
    value.map_or("", |s| s.trim()).to_owned()
}

/// Render a single embed; `""` when nothing meaningful is present.
fn format_one(embed: &EmbedView) -> String {
    let provider_name = stripped(embed.provider_name.as_ref());
    let author_name = stripped(embed.author_name.as_ref());
    let title = stripped(embed.title.as_ref());
    let description = stripped(embed.description.as_ref());
    let footer_text = stripped(embed.footer_text.as_ref());
    let url = stripped(embed.url.as_ref());

    let mut header_bits: Vec<String> = Vec::new();
    if !provider_name.is_empty() {
        header_bits.push(format!("({provider_name})"));
    }
    if !author_name.is_empty() {
        header_bits.push(author_name.clone());
    }
    // Avoid echoing the same string twice when the title mirrors the author
    // handle (common on Tumblr / Bluesky cards).
    if !title.is_empty() && title != author_name {
        header_bits.push(title);
    }
    let header = header_bits.join(" \u{2014} ");

    let mut lines: Vec<String> = Vec::new();
    if !header.is_empty() {
        lines.push(header);
    }
    if !description.is_empty() {
        lines.push(description);
    }

    for field in &embed.fields {
        let name = stripped(field.name.as_ref());
        let value = stripped(field.value.as_ref());
        if !name.is_empty() && !value.is_empty() {
            lines.push(format!("{name}: {value}"));
        } else if !name.is_empty() || !value.is_empty() {
            lines.push(if name.is_empty() { value } else { name });
        }
    }

    if !footer_text.is_empty() {
        lines.push(format!("\u{2014} {footer_text}"));
    }

    if lines.is_empty() {
        // Image-only embed: surface the link target so the LLM at least knows a
        // media URL was attached. Drop entirely when even the url is missing.
        if url.is_empty() {
            return String::new();
        }
        return format!("{EMBED_TAG}\n[link: {url}]");
    }

    format!("{EMBED_TAG}\n{}", lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::{EmbedFieldView, EmbedView, format_embeds};

    fn desc(text: &str) -> EmbedView {
        EmbedView {
            description: Some(text.to_owned()),
            ..Default::default()
        }
    }

    #[test]
    fn empty_iterable_returns_empty_string() {
        assert!(format_embeds(&[]).is_empty());
    }

    #[test]
    fn skips_blank_embeds() {
        // An embed with no text fields contributes nothing.
        assert!(format_embeds(&[EmbedView::default()]).is_empty());
    }

    #[test]
    fn description_only_embed() {
        assert_eq!(
            format_embeds(&[desc("hello world")]),
            "[embed]\nhello world"
        );
    }

    #[test]
    fn renders_provider_author_title_description() {
        let e = EmbedView {
            title: Some("The beach that makes you old".to_owned()),
            description: Some("body".to_owned()),
            author_name: Some("dotshaft".to_owned()),
            provider_name: Some("Tumblr".to_owned()),
            ..Default::default()
        };
        assert_eq!(
            format_embeds(&[e]),
            "[embed]\n(Tumblr) \u{2014} dotshaft \u{2014} The beach that makes you old\nbody"
        );
    }

    #[test]
    fn dedupes_title_when_equal_to_author() {
        let e = EmbedView {
            title: Some("dotshaft".to_owned()),
            description: Some("body".to_owned()),
            author_name: Some("dotshaft".to_owned()),
            ..Default::default()
        };
        // title repeats author -> drop title from header
        assert_eq!(format_embeds(&[e]), "[embed]\ndotshaft\nbody");
    }

    #[test]
    fn renders_fields() {
        let e = EmbedView {
            title: Some("match".to_owned()),
            fields: vec![
                EmbedFieldView {
                    name: Some("Score".to_owned()),
                    value: Some("3-1".to_owned()),
                },
                EmbedFieldView {
                    name: Some("Stadium".to_owned()),
                    value: Some("Anfield".to_owned()),
                },
            ],
            ..Default::default()
        };
        assert_eq!(
            format_embeds(&[e]),
            "[embed]\nmatch\nScore: 3-1\nStadium: Anfield"
        );
    }

    #[test]
    fn renders_footer() {
        let e = EmbedView {
            description: Some("body".to_owned()),
            footer_text: Some("via example.com".to_owned()),
            ..Default::default()
        };
        assert_eq!(
            format_embeds(&[e]),
            "[embed]\nbody\n\u{2014} via example.com"
        );
    }

    #[test]
    fn image_only_embed_falls_back_to_url() {
        let e = EmbedView {
            url: Some("https://example.com/x".to_owned()),
            ..Default::default()
        };
        assert_eq!(
            format_embeds(&[e]),
            "[embed]\n[link: https://example.com/x]"
        );
    }

    #[test]
    fn image_only_embed_without_url_returns_empty() {
        assert!(format_embeds(&[EmbedView::default()]).is_empty());
    }

    #[test]
    fn multiple_embeds_separated_by_blank_line() {
        let out = format_embeds(&[desc("first"), desc("second")]);
        assert_eq!(out, "[embed]\nfirst\n\n[embed]\nsecond");
    }

    #[test]
    fn tumblr_reblog_chain_renders_verbatim() {
        // Discord's Tumblr unfurl puts the entire reblog chain in the
        // description verbatim; the formatter must not reflow it.
        let chain = "\u{1f501} dotshaft\n\n\
             shittymoviedetails\n\n\
             The beach that makes you old\n\n\
             powerjock\n\n\
             I can never seem to find her, but she always finds me";
        let e = EmbedView {
            description: Some(chain.to_owned()),
            author_name: Some("fratal".to_owned()),
            provider_name: Some("Tumblr".to_owned()),
            ..Default::default()
        };
        assert_eq!(
            format_embeds(&[e]),
            format!("[embed]\n(Tumblr) \u{2014} fratal\n{chain}")
        );
    }

    #[test]
    fn duck_typed_input() {
        let e = EmbedView {
            title: Some("t".to_owned()),
            description: Some("d".to_owned()),
            author_name: Some("a".to_owned()),
            provider_name: Some("p".to_owned()),
            fields: vec![EmbedFieldView {
                name: Some("k".to_owned()),
                value: Some("v".to_owned()),
            }],
            footer_text: Some("f".to_owned()),
            ..Default::default()
        };
        assert_eq!(
            format_embeds(&[e]),
            "[embed]\n(p) \u{2014} a \u{2014} t\nd\nk: v\n\u{2014} f"
        );
    }
}
