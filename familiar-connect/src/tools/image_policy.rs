//! Fetch-boundary URL policy for `view_image` (subsystem 08).
//!
//! Two gates, both applied before any request is issued:
//!
//! 1. **Unconditional** — http(s) only, and every resolved address must be
//!    public unicast. Loopback, RFC1918, link-local (incl. the
//!    `169.254.169.254` cloud-metadata address), CGNAT, and their IPv6
//!    equivalents are refused with no config escape.
//! 2. **Host allowlist** — [`ImageUrlPolicy::trusted_hosts`], bypassed by
//!    `allow_untrusted`.
//!
//! Enforced at the fetch boundary rather than at image collection, so all
//! three image sources (attachments, embeds, regex-scraped inline URLs in
//! message text) pass through one check and a fourth source cannot bypass it.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;

use async_trait::async_trait;

/// Hosts trusted with no config change.
pub const DEFAULT_TRUSTED_IMAGE_HOSTS: [&str; 12] = [
    "cdn.discordapp.com",
    "media.discordapp.net",
    "images-ext-1.discordapp.net",
    "images-ext-2.discordapp.net",
    "*.media.tumblr.com",
    "i.imgur.com",
    "media.tenor.com",
    "c.tenor.com",
    "media.giphy.com",
    "i.giphy.com",
    "i.redd.it",
    "preview.redd.it",
];

/// Host-allowlist gate for image fetches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageUrlPolicy {
    /// Bypass the allowlist (rule 1 still applies).
    pub allow_untrusted: bool,
    /// Exact hostnames, or `*.suffix` patterns matching any subdomain.
    pub trusted_hosts: Vec<String>,
}

impl Default for ImageUrlPolicy {
    fn default() -> Self {
        Self {
            allow_untrusted: false,
            trusted_hosts: DEFAULT_TRUSTED_IMAGE_HOSTS
                .iter()
                .map(|h| (*h).to_owned())
                .collect(),
        }
    }
}

impl ImageUrlPolicy {
    /// Build from the parsed `[tools]` section.
    #[must_use]
    pub fn from_tools_config(cfg: &crate::config::ToolsConfig) -> Self {
        Self {
            allow_untrusted: cfg.allow_untrusted_image_urls,
            trusted_hosts: cfg.trusted_image_hosts.clone(),
        }
    }
}

/// Why a URL was refused. Messages are test contracts.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum UrlRefusal {
    /// Unparseable, or carrying control/whitespace characters.
    #[error("malformed image url")]
    Malformed,
    /// Scheme other than http/https.
    #[error("image url scheme '{0}' is not allowed — only http/https")]
    Scheme(String),
    /// Host absent from the allowlist while `allow_untrusted` is false.
    #[error(
        "image host '{0}' is not in [tools].trusted_image_hosts — set [tools].allow_untrusted_image_urls = true to allow it"
    )]
    UntrustedHost(String),
    /// DNS returned nothing (or failed).
    #[error("image host '{0}' did not resolve")]
    Unresolvable(String),
    /// Resolved (or literal) address is not public unicast.
    #[error("image host '{host}' resolves to non-public address {addr}")]
    NonPublicAddress {
        /// Hostname as written in the URL.
        host: String,
        /// The offending address.
        addr: IpAddr,
    },
}

/// Scheme / host / port lifted out of a URL.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UrlParts {
    /// Lowercased scheme (`http` or `https`).
    pub scheme: String,
    /// Lowercased host, trailing root dot stripped, brackets removed.
    pub host: String,
    /// Explicit port, else the scheme default.
    pub port: u16,
}

/// A URL that cleared both gates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckedUrl {
    /// Normalized host.
    pub host: String,
    /// Effective port.
    pub port: u16,
    /// Validated addresses — pin the connection to these.
    pub addrs: Vec<SocketAddr>,
}

/// DNS seam so tests never touch the network.
#[async_trait]
pub trait HostResolver: Send + Sync {
    /// Resolve `host`; `port` only shapes the lookup.
    async fn resolve(&self, host: &str, port: u16) -> anyhow::Result<Vec<IpAddr>>;
}

/// Production resolver (the OS resolver, via tokio).
pub struct SystemResolver;

#[async_trait]
impl HostResolver for SystemResolver {
    async fn resolve(&self, host: &str, port: u16) -> anyhow::Result<Vec<IpAddr>> {
        Ok(tokio::net::lookup_host((host, port))
            .await?
            .map(|sa| sa.ip())
            .collect())
    }
}

/// Lowercase, strip IPv6 brackets and the trailing root dot.
#[must_use]
pub fn normalize_host(host: &str) -> String {
    host.trim_start_matches('[')
        .trim_end_matches(']')
        .trim_end_matches('.')
        .to_ascii_lowercase()
}

/// Whether every label of `s` is a plausible DNS label.
fn is_domain(s: &str) -> bool {
    !s.is_empty()
        && s.split('.').all(|label| {
            !label.is_empty()
                && label
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        })
}

/// Whether `s` is a bare hostname or `*.`-prefixed suffix pattern.
#[must_use]
pub fn is_host_pattern(s: &str) -> bool {
    is_domain(s.strip_prefix("*.").unwrap_or(s))
}

/// Whether `host` matches an allowlist entry.
///
/// Exact match, or `*.suffix` against any strict subdomain of `suffix` — the
/// bare suffix itself does not match, nor does a name merely ending in the
/// suffix text.
#[must_use]
pub fn host_is_trusted(host: &str, trusted: &[String]) -> bool {
    trusted.iter().any(|entry| {
        entry.strip_prefix("*.").map_or_else(
            || entry == host,
            |suffix| host.len() > suffix.len() + 1 && host.ends_with(&format!(".{suffix}")),
        )
    })
}

/// Whether `ip` is a public unicast address.
#[must_use]
pub const fn is_public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_public_v4(v4),
        IpAddr::V6(v6) => is_public_v6(v6),
    }
}

/// Reject every IPv4 range that is not globally routable unicast.
const fn is_public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(a == 0                                  // 0.0.0.0/8 "this network"
        || ip.is_loopback()                   // 127/8
        || ip.is_private()                    // 10/8, 172.16/12, 192.168/16
        || ip.is_link_local()                 // 169.254/16, incl. cloud metadata
        || ip.is_multicast()                  // 224/4
        || ip.is_broadcast()
        || ip.is_documentation()              // 192.0.2/24, 198.51.100/24, 203.0.113/24
        || (a == 100 && b >= 64 && b < 128)   // CGNAT 100.64/10
        || (a == 192 && b == 0 && c == 0)     // IETF protocol assignments 192.0.0/24
        || (a == 198 && (b == 18 || b == 19)) // benchmarking 198.18/15
        || a >= 240) // reserved 240/4
}

/// Same, for IPv6 — v4-bearing forms are judged by their embedded v4.
const fn is_public_v6(ip: Ipv6Addr) -> bool {
    let o = ip.octets();
    // ::ffff:a.b.c.d (v4-mapped), ::a.b.c.d (v4-compatible), 64:ff9b::/96 (NAT64)
    if let Some(v4) = embedded_v4(o) {
        return is_public_v4(v4);
    }
    !(ip.is_unspecified()
        || ip.is_loopback()
        || ip.is_multicast()
        || (o[0] & 0xfe) == 0xfc                                    // ULA fc00::/7
        || (o[0] == 0xfe && (o[1] & 0xc0) == 0x80)                  // link-local fe80::/10
        || (o[0] == 0x20 && o[1] == 0x01 && o[2] == 0x0d && o[3] == 0xb8)) // doc 2001:db8::/32
}

/// The IPv4 address an IPv6 address carries, if any.
const fn embedded_v4(o: [u8; 16]) -> Option<Ipv4Addr> {
    let leading_zeros = o[0] == 0
        && o[1] == 0
        && o[2] == 0
        && o[3] == 0
        && o[4] == 0
        && o[5] == 0
        && o[6] == 0
        && o[7] == 0
        && o[8] == 0
        && o[9] == 0;
    let mapped = leading_zeros && o[10] == 0xff && o[11] == 0xff;
    let compatible = leading_zeros && o[10] == 0 && o[11] == 0;
    let nat64 = o[0] == 0
        && o[1] == 0x64
        && o[2] == 0xff
        && o[3] == 0x9b
        && o[4] == 0
        && o[5] == 0
        && o[6] == 0
        && o[7] == 0
        && o[8] == 0
        && o[9] == 0
        && o[10] == 0
        && o[11] == 0;
    if mapped || compatible || nat64 {
        Some(Ipv4Addr::new(o[12], o[13], o[14], o[15]))
    } else {
        None
    }
}

/// Split a URL into scheme / host / port.
///
/// Deliberately strict: anything carrying whitespace, control characters, or a
/// backslash is refused rather than guessed at, so this cannot disagree with
/// the WHATWG parser `reqwest` uses (the fetcher cross-checks the host it ends
/// up with anyway).
pub fn split_url(url: &str) -> Result<UrlParts, UrlRefusal> {
    if url.is_empty()
        || url
            .chars()
            .any(|c| c.is_whitespace() || c.is_control() || c == '\\')
    {
        return Err(UrlRefusal::Malformed);
    }
    let (scheme, rest) = url.split_once(':').ok_or(UrlRefusal::Malformed)?;
    if !scheme.starts_with(|c: char| c.is_ascii_alphabetic())
        || !scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    {
        return Err(UrlRefusal::Malformed);
    }
    let scheme = scheme.to_ascii_lowercase();
    let default_port = match scheme.as_str() {
        "http" => 80,
        "https" => 443,
        _ => return Err(UrlRefusal::Scheme(scheme)),
    };

    let rest = rest.strip_prefix("//").ok_or(UrlRefusal::Malformed)?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    // Userinfo before the last '@' is not the host.
    let authority = authority.rsplit('@').next().unwrap_or("");

    let (host, port) = if let Some(after_bracket) = authority.strip_prefix('[') {
        let (host, tail) = after_bracket.split_once(']').ok_or(UrlRefusal::Malformed)?;
        (host, parse_port(tail.strip_prefix(':'), default_port)?)
    } else if let Some((host, port)) = authority.rsplit_once(':') {
        (host, parse_port(Some(port), default_port)?)
    } else {
        (authority, default_port)
    };

    let host = normalize_host(host);
    if host.is_empty() {
        return Err(UrlRefusal::Malformed);
    }
    Ok(UrlParts { scheme, host, port })
}

fn parse_port(raw: Option<&str>, default_port: u16) -> Result<u16, UrlRefusal> {
    raw.map_or(Ok(default_port), |p| {
        p.parse().map_err(|_| UrlRefusal::Malformed)
    })
}

/// Policy + resolver, applied to every URL at the fetch boundary.
pub struct UrlGuard {
    policy: ImageUrlPolicy,
    resolver: Arc<dyn HostResolver>,
}

impl UrlGuard {
    /// Guard with an injected resolver.
    #[must_use]
    pub fn new(policy: ImageUrlPolicy, resolver: Arc<dyn HostResolver>) -> Self {
        Self { policy, resolver }
    }

    /// Guard using the OS resolver.
    #[must_use]
    pub fn production(policy: ImageUrlPolicy) -> Self {
        Self::new(policy, Arc::new(SystemResolver))
    }

    /// Refuse or admit `url`; on admit, return the addresses to pin to.
    pub async fn check(&self, url: &str) -> Result<CheckedUrl, UrlRefusal> {
        let UrlParts { host, port, .. } = split_url(url)?;
        if !self.policy.allow_untrusted && !host_is_trusted(&host, &self.policy.trusted_hosts) {
            return Err(UrlRefusal::UntrustedHost(host));
        }
        // Rule 1 runs on the *resolved* addresses, so a name pointing at
        // 127.0.0.1 is refused like the literal — allowlist or flag regardless.
        let addrs = match host.parse::<IpAddr>() {
            Ok(literal) => vec![literal],
            Err(_) => self
                .resolver
                .resolve(&host, port)
                .await
                .map_err(|_| UrlRefusal::Unresolvable(host.clone()))?,
        };
        if addrs.is_empty() {
            return Err(UrlRefusal::Unresolvable(host));
        }
        for addr in &addrs {
            if !is_public_ip(*addr) {
                return Err(UrlRefusal::NonPublicAddress { host, addr: *addr });
            }
        }
        Ok(CheckedUrl {
            addrs: addrs
                .into_iter()
                .map(|ip| SocketAddr::new(ip, port))
                .collect(),
            host,
            port,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse().expect("test address")
    }

    #[test]
    fn default_policy_denies_untrusted() {
        assert!(!ImageUrlPolicy::default().allow_untrusted);
    }

    #[test]
    fn discord_hosts_are_trusted_by_default() {
        let d = ImageUrlPolicy::default();
        for host in [
            "cdn.discordapp.com",
            "media.discordapp.net",
            "images-ext-1.discordapp.net",
        ] {
            assert!(host_is_trusted(host, &d.trusted_hosts), "{host}");
        }
    }

    #[test]
    fn tumblr_shards_match_the_wildcard_entry() {
        let d = ImageUrlPolicy::default();
        assert!(host_is_trusted("64.media.tumblr.com", &d.trusted_hosts));
        assert!(host_is_trusted("44.media.tumblr.com", &d.trusted_hosts));
        // the bare suffix and look-alikes must not match
        assert!(!host_is_trusted("media.tumblr.com", &d.trusted_hosts));
        assert!(!host_is_trusted("evilmedia.tumblr.com", &d.trusted_hosts));
        assert!(!host_is_trusted(
            "64.media.tumblr.com.evil.test",
            &d.trusted_hosts
        ));
    }

    #[test]
    fn untrusted_host_not_matched() {
        let d = ImageUrlPolicy::default();
        assert!(!host_is_trusted("attacker.example", &d.trusted_hosts));
    }

    #[test]
    fn split_url_reads_scheme_host_port() {
        let p = split_url("https://cdn.discordapp.com/a/b.png?x=1").expect("parses");
        assert_eq!(p.scheme, "https");
        assert_eq!(p.host, "cdn.discordapp.com");
        assert_eq!(p.port, 443);
        assert_eq!(split_url("http://h.test/x").expect("parses").port, 80);
        assert_eq!(
            split_url("http://h.test:8080/x").expect("parses").port,
            8080
        );
    }

    #[test]
    fn split_url_normalizes_case_userinfo_and_root_dot() {
        // userinfo must not be mistaken for the host
        assert_eq!(
            split_url("http://cdn.discordapp.com@attacker.example/x.png")
                .expect("parses")
                .host,
            "attacker.example"
        );
        assert_eq!(
            split_url("HTTPS://CDN.DiscordApp.COM./x.png")
                .expect("parses")
                .host,
            "cdn.discordapp.com"
        );
        // fragment terminates the authority
        assert_eq!(
            split_url("https://cdn.discordapp.com#@attacker.example")
                .expect("parses")
                .host,
            "cdn.discordapp.com"
        );
    }

    #[test]
    fn split_url_rejects_non_http_schemes() {
        assert_eq!(
            split_url("file:///etc/passwd"),
            Err(UrlRefusal::Scheme("file".to_owned()))
        );
        assert_eq!(
            split_url("data:image/png;base64,AAAA"),
            Err(UrlRefusal::Scheme("data".to_owned()))
        );
        assert_eq!(
            split_url("gopher://h.test/x.png"),
            Err(UrlRefusal::Scheme("gopher".to_owned()))
        );
    }

    #[test]
    fn split_url_rejects_junk() {
        for bad in [
            "",
            "not a url",
            "https://",
            "https:/h.test/x",
            "https://h.test\\@attacker.example/x",
            "https://h\ttest/x",
            "https://h.test:80a/x",
        ] {
            assert!(split_url(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn split_url_handles_bracketed_ipv6() {
        let p = split_url("http://[::1]:8080/x.png").expect("parses");
        assert_eq!(p.host, "::1");
        assert_eq!(p.port, 8080);
    }

    #[test]
    fn private_and_reserved_v4_is_not_public() {
        for bad in [
            "127.0.0.1",
            "127.5.5.5",
            "0.0.0.0",
            "10.0.0.5",
            "172.16.3.4",
            "192.168.1.1",
            "169.254.169.254", // cloud metadata
            "169.254.0.1",
            "100.64.0.1", // CGNAT
            "192.0.0.1",  // IETF protocol assignments
            "198.18.0.1", // benchmarking
            "192.0.2.5",  // documentation
            "224.0.0.1",  // multicast
            "255.255.255.255",
            "240.0.0.1",
        ] {
            assert!(!is_public_ip(ip(bad)), "{bad} must not be public");
        }
    }

    #[test]
    fn public_v4_is_public() {
        for good in ["1.1.1.1", "162.159.135.232", "8.8.8.8", "99.83.1.1"] {
            assert!(is_public_ip(ip(good)), "{good} must be public");
        }
    }

    #[test]
    fn private_and_reserved_v6_is_not_public() {
        for bad in [
            "::1",
            "::",
            "fc00::1",
            "fd12:3456::1",
            "fe80::1",
            "ff02::1",
            "2001:db8::1",
            "::ffff:127.0.0.1", // v4-mapped loopback
            "::ffff:10.0.0.1",
            "::127.0.0.1",     // v4-compatible
            "64:ff9b::7f00:1", // NAT64-embedded loopback
        ] {
            assert!(!is_public_ip(ip(bad)), "{bad} must not be public");
        }
    }

    #[test]
    fn public_v6_is_public() {
        assert!(is_public_ip(ip("2606:4700::1111")));
        assert!(is_public_ip(ip("::ffff:1.1.1.1")));
    }

    /// Canned DNS: every host resolves to the given addresses.
    struct FakeResolver {
        addrs: Vec<IpAddr>,
    }

    #[async_trait]
    impl HostResolver for FakeResolver {
        async fn resolve(&self, _host: &str, _port: u16) -> anyhow::Result<Vec<IpAddr>> {
            if self.addrs.is_empty() {
                anyhow::bail!("no such host");
            }
            Ok(self.addrs.clone())
        }
    }

    fn guard(allow_untrusted: bool, resolves_to: &[&str]) -> UrlGuard {
        UrlGuard::new(
            ImageUrlPolicy {
                allow_untrusted,
                ..ImageUrlPolicy::default()
            },
            Arc::new(FakeResolver {
                addrs: resolves_to.iter().map(|s| ip(s)).collect(),
            }),
        )
    }

    #[tokio::test]
    async fn trusted_host_passes_by_default() {
        let checked = guard(false, &["162.159.135.232"])
            .check("https://cdn.discordapp.com/attachments/1/2/cat.png")
            .await
            .expect("discord cdn passes");
        assert_eq!(checked.host, "cdn.discordapp.com");
        assert_eq!(checked.port, 443);
        assert_eq!(
            checked.addrs,
            vec![SocketAddr::new(ip("162.159.135.232"), 443)]
        );
    }

    #[tokio::test]
    async fn untrusted_host_refused_by_default() {
        assert_eq!(
            guard(false, &["93.184.216.34"])
                .check("https://attacker.example/x.png")
                .await,
            Err(UrlRefusal::UntrustedHost("attacker.example".to_owned()))
        );
    }

    #[tokio::test]
    async fn untrusted_host_allowed_when_flag_set() {
        assert!(
            guard(true, &["93.184.216.34"])
                .check("https://attacker.example/x.png")
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn private_literal_refused_even_when_flag_set() {
        for (url, addr) in [
            ("http://127.0.0.1/x.png", "127.0.0.1"),
            ("http://169.254.169.254/latest/meta-data", "169.254.169.254"),
            ("http://10.1.2.3/x.png", "10.1.2.3"),
            ("http://[::1]/x.png", "::1"),
        ] {
            assert_eq!(
                guard(true, &[]).check(url).await,
                Err(UrlRefusal::NonPublicAddress {
                    host: ip(addr).to_string(),
                    addr: ip(addr),
                }),
                "{url}"
            );
        }
    }

    #[tokio::test]
    async fn hostname_resolving_to_private_refused_even_when_flag_set() {
        assert_eq!(
            guard(true, &["127.0.0.1"])
                .check("https://rebind.example/x.png")
                .await,
            Err(UrlRefusal::NonPublicAddress {
                host: "rebind.example".to_owned(),
                addr: ip("127.0.0.1"),
            })
        );
    }

    #[tokio::test]
    async fn one_private_address_among_many_refuses_the_host() {
        assert_eq!(
            guard(true, &["1.1.1.1", "192.168.0.9"])
                .check("https://mixed.example/x.png")
                .await,
            Err(UrlRefusal::NonPublicAddress {
                host: "mixed.example".to_owned(),
                addr: ip("192.168.0.9"),
            })
        );
    }

    #[tokio::test]
    async fn trusted_host_resolving_to_private_still_refused() {
        assert_eq!(
            guard(false, &["127.0.0.1"])
                .check("https://cdn.discordapp.com/x.png")
                .await,
            Err(UrlRefusal::NonPublicAddress {
                host: "cdn.discordapp.com".to_owned(),
                addr: ip("127.0.0.1"),
            })
        );
    }

    #[tokio::test]
    async fn unresolvable_host_refused() {
        assert_eq!(
            guard(true, &[])
                .check("https://nowhere.example/x.png")
                .await,
            Err(UrlRefusal::Unresolvable("nowhere.example".to_owned()))
        );
    }

    #[tokio::test]
    async fn non_http_scheme_refused_even_when_flag_set() {
        assert_eq!(
            guard(true, &["1.1.1.1"]).check("file:///etc/passwd").await,
            Err(UrlRefusal::Scheme("file".to_owned()))
        );
    }

    #[test]
    fn host_pattern_validation() {
        for ok in ["cdn.discordapp.com", "*.media.tumblr.com", "localhost"] {
            assert!(is_host_pattern(ok), "{ok}");
        }
        for bad in [
            "",
            "*",
            "*.",
            "https://cdn.discordapp.com",
            "cdn.discordapp.com/path",
            "cdn.discordapp.com:443",
            "cdn discordapp com",
            ".cdn.discordapp.com",
            "cdn..discordapp.com",
            "cdn.discordapp.com.",
            "a.*.com",
        ] {
            assert!(!is_host_pattern(bad), "{bad:?} should be rejected");
        }
    }
}
