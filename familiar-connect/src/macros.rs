//! Minimal SillyTavern macro substitution.
//!
//! Port of `familiar_connect/macros.py`. Supported macros (deliberate subset —
//! unknown macros pass through unchanged): `{{char}}`, `{{user}}` (default
//! `"User"`), `{{trim}}`, `{{scenario}}`, `{{personality}}`, `{{description}}`,
//! and `{{// ... }}` comments (removed entirely).
//!
//! Currently dead code upstream — only `tests/test_macros.py` references it. The
//! module is `pub`; nothing wires it yet (identity/character-card code in
//! subsystem 02 is its intended consumer).

use std::sync::LazyLock;

use regex::{Captures, Regex};

/// Values substituted into macro placeholders.
#[derive(Clone, Debug)]
pub struct MacroContext {
    /// `{{char}}` — character name (default empty).
    pub char: String,
    /// `{{user}}` — user name (default `"User"`).
    pub user: String,
    /// `{{scenario}}` — character scenario (default empty).
    pub scenario: String,
    /// `{{personality}}` — character personality (default empty).
    pub personality: String,
    /// `{{description}}` — character description (default empty).
    pub description: String,
}

impl Default for MacroContext {
    fn default() -> Self {
        Self {
            char: String::new(),
            user: "User".to_string(),
            scenario: String::new(),
            personality: String::new(),
            description: String::new(),
        }
    }
}

// Matches `{{// any comment text }}` (comment body cannot contain `}`).
static COMMENT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{\{//[^}]*\}\}").expect("valid comment regex"));

// Matches any `{{macro}}` or `{{macro::arg}}` token.
static MACRO_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{\{([^}]+)\}\}").expect("valid macro regex"));

/// Resolve macros in `text` using `ctx`.
///
/// Order: (1) strip comments; (2) single pass over `{{...}}` tokens — `{{trim}}`
/// sets a flag and expands to `""`, known simple macros expand to context
/// values, anything else passes through verbatim; (3) if any `{{trim}}` was
/// seen, strip whitespace from the whole result. Single pass — substituted
/// values are never re-scanned.
#[must_use]
pub fn substitute(text: &str, ctx: &MacroContext) -> String {
    // 1. strip comments
    let stripped = COMMENT_RE.replace_all(text, "");

    // 2 & 3. replace all macros in one pass
    let mut trim_requested = false;
    let replaced = MACRO_RE.replace_all(&stripped, |caps: &Captures| {
        let key = caps[1].trim();
        match key {
            "trim" => {
                trim_requested = true;
                String::new()
            }
            "char" => ctx.char.clone(),
            "user" => ctx.user.clone(),
            "scenario" => ctx.scenario.clone(),
            "personality" => ctx.personality.clone(),
            "description" => ctx.description.clone(),
            // Unknown macro — pass through verbatim.
            _ => caps[0].to_string(),
        }
    });

    // 4. apply trim if requested
    if trim_requested {
        replaced.trim().to_string()
    } else {
        replaced.into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::{MacroContext, substitute};

    fn ctx() -> MacroContext {
        MacroContext::default()
    }

    // --- comment macro ---

    #[test]
    fn comment_is_removed() {
        assert_eq!(
            substitute("Hello {{// this is a comment }}world", &ctx()),
            "Hello world"
        );
    }

    #[test]
    fn multiword_comment_removed() {
        assert!(substitute("{{// remove me }}", &ctx()).is_empty());
    }

    #[test]
    fn comment_with_spaces() {
        assert_eq!(
            substitute("a{{// comment with lots of words here }}b", &ctx()),
            "ab"
        );
    }

    // --- trim macro ---

    #[test]
    fn trim_removes_macro_and_strips_whitespace() {
        assert_eq!(substitute("  hello world  {{trim}}", &ctx()), "hello world");
    }

    #[test]
    fn trim_with_leading_whitespace() {
        assert_eq!(substitute("  {{trim}}  text  ", &ctx()), "text");
    }

    #[test]
    fn trim_only() {
        assert!(substitute("{{trim}}", &ctx()).is_empty());
    }

    // --- simple macros + defaults ---

    #[test]
    fn char_replaced_with_name() {
        let c = MacroContext {
            char: "Sapphire".to_string(),
            ..MacroContext::default()
        };
        assert_eq!(
            substitute("My name is {{char}}.", &c),
            "My name is Sapphire."
        );
    }

    #[test]
    fn char_default_empty() {
        assert!(substitute("{{char}}", &ctx()).is_empty());
    }

    #[test]
    fn user_replaced_with_name() {
        let c = MacroContext {
            user: "Alice".to_string(),
            ..MacroContext::default()
        };
        assert_eq!(substitute("Hello, {{user}}!", &c), "Hello, Alice!");
    }

    #[test]
    fn user_default() {
        assert_eq!(substitute("{{user}}", &ctx()), "User");
    }

    #[test]
    fn scenario_substituted() {
        let c = MacroContext {
            scenario: "A rainy afternoon in a coffee shop.".to_string(),
            ..MacroContext::default()
        };
        assert_eq!(
            substitute("Scene: {{scenario}}", &c),
            "Scene: A rainy afternoon in a coffee shop."
        );
    }

    #[test]
    fn scenario_default_empty() {
        assert!(substitute("{{scenario}}", &ctx()).is_empty());
    }

    #[test]
    fn personality_substituted() {
        let c = MacroContext {
            personality: "Cheerful and curious.".to_string(),
            ..MacroContext::default()
        };
        assert_eq!(
            substitute("Traits: {{personality}}", &c),
            "Traits: Cheerful and curious."
        );
    }

    #[test]
    fn personality_default_empty() {
        assert!(substitute("{{personality}}", &ctx()).is_empty());
    }

    #[test]
    fn description_substituted() {
        let c = MacroContext {
            description: "A blue-haired mage.".to_string(),
            ..MacroContext::default()
        };
        assert_eq!(
            substitute("Who: {{description}}", &c),
            "Who: A blue-haired mage."
        );
    }

    #[test]
    fn description_default_empty() {
        assert!(substitute("{{description}}", &ctx()).is_empty());
    }

    // --- unknown macros pass through ---

    #[test]
    fn unknown_macro_passes_through() {
        assert_eq!(
            substitute("{{getvar::guidelines}}", &ctx()),
            "{{getvar::guidelines}}"
        );
    }

    #[test]
    fn unknown_macro_with_text() {
        assert_eq!(
            substitute("before {{randomfuture}} after", &ctx()),
            "before {{randomfuture}} after"
        );
    }

    // --- combined pipeline order ---

    #[test]
    fn multiple_macros_in_one_string() {
        let c = MacroContext {
            char: "Sapphire".to_string(),
            user: "Alice".to_string(),
            personality: "Bright".to_string(),
            ..MacroContext::default()
        };
        let text = "{{// header }}{{char}} speaks to {{user}}. Traits: {{personality}}{{trim}}";
        assert_eq!(
            substitute(text, &c),
            "Sapphire speaks to Alice. Traits: Bright"
        );
    }

    #[test]
    fn comment_then_trim() {
        assert_eq!(
            substitute("  {{// ignore }}  hello  {{trim}}", &ctx()),
            "hello"
        );
    }
}
