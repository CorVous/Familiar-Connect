//! Human-paced typing hold for text replies (subsystem 06).
//!
//! A reply that streams in fast would otherwise land a moment after "is
//! typing…" appears. [`TypingPace`] turns a reply's length into how long the
//! indicator should have been up before it posts; the
//! [`TextResponder`](crate::processors::text_responder::TextResponder) holds
//! the send for whatever is left after generation.

use tokio::time::Duration;

use crate::config::DiscordTextConfig;

/// Re-broadcast interval; one Discord typing broadcast lasts ~10 s.
pub const TYPING_REFRESH: Duration = Duration::from_secs(8);

/// Reply length → typing duration (`[discord.text].typing_*`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TypingPace {
    /// Characters "typed" per second.
    pub chars_per_second: f64,
    /// Floor, so a one-word reply still shows a beat of typing.
    pub min: Duration,
    /// Ceiling, so a long reply never stalls.
    pub max: Duration,
}

impl TypingPace {
    /// Pace from config; `None` when `typing_delay` is off.
    #[must_use]
    pub fn from_config(cfg: &DiscordTextConfig) -> Option<Self> {
        cfg.typing_delay.then(|| Self {
            chars_per_second: cfg.typing_chars_per_second,
            min: Duration::from_secs_f64(cfg.typing_delay_min_s),
            max: Duration::from_secs_f64(cfg.typing_delay_max_s),
        })
    }

    /// Total typing time a reply of `chars` characters should show:
    /// `chars / chars_per_second`, clamped to `[min, max]`.
    #[must_use]
    pub fn duration_for(&self, chars: usize) -> Duration {
        #[allow(clippy::cast_precision_loss)] // Reply lengths are far below 2^52.
        let secs = chars as f64 / self.chars_per_second;
        let raw = Duration::try_from_secs_f64(secs).unwrap_or(self.max);
        raw.clamp(self.min, self.max.max(self.min))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pace() -> TypingPace {
        TypingPace {
            chars_per_second: 10.0,
            min: Duration::from_secs(1),
            max: Duration::from_secs(6),
        }
    }

    #[test]
    fn scales_with_length() {
        assert_eq!(pace().duration_for(30), Duration::from_secs(3));
    }

    #[test]
    fn short_reply_gets_floor() {
        assert_eq!(pace().duration_for(2), Duration::from_secs(1));
        assert_eq!(pace().duration_for(0), Duration::from_secs(1));
    }

    #[test]
    fn long_reply_capped() {
        assert_eq!(pace().duration_for(5_000), Duration::from_secs(6));
    }

    #[test]
    fn disabled_config_yields_none() {
        let cfg = DiscordTextConfig {
            typing_delay: false,
            ..DiscordTextConfig::default()
        };
        assert_eq!(TypingPace::from_config(&cfg), None);
    }

    #[test]
    fn default_config_enabled() {
        let p = TypingPace::from_config(&DiscordTextConfig::default()).expect("on by default");
        assert!(p.chars_per_second > 0.0);
        assert!(p.min <= p.max);
    }
}
