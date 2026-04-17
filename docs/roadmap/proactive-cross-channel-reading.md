# Proactive cross-channel reading

When the familiar sees a `https://discord.com/channels/<g>/<c>/<m>`
link pointing at an **accessible-but-unsubscribed** channel, should
it be allowed to follow the link and read the referenced message?

!!! info "Status: Deferred"
    Today the bot resolves the *channel name* for accessible channels
    but only fetches the *message body* for channels the operator has
    subscribed the familiar to. This page tracks the question of
    whether to relax that.

## Motivation

Two distinct axes already exist (see [Message flow § Subscribed vs.
accessible channels](../architecture/message-flow.md#subscribed-vs-accessible-channels)):

- **Subscribed** — operator opt-in. Familiar actively listens and
  replies. Small set, curated.
- **Accessible** — bot's Discord role can `read_messages`. A superset.

The current rule resolves the channel name for either bucket (the
sender already sees that name in the `<#id>` token Discord renders),
but only pulls the message body when the target is in the subscribed
bucket. That keeps the familiar's reading surface equal to what the
operator has explicitly consented to.

The successor question: is it useful for the familiar to be able to
"peek" at a linked message in a channel it's not subscribed to, just
because someone pasted the link? The motivation is mostly answering
questions like "what did Alice mean here?" without the operator
having to add every channel the familiar might be asked about.

## Sketch

Two plausible designs:

- **Read-only scope.** A separate subscription kind
  (`SubscriptionKind.read_only`?) — the familiar resolves message
  bodies from these channels but never posts to them, never runs its
  monitor over them, never summarises them into history. Stays
  bounded and explicit.
- **Per-link policy.** Keep subscriptions as-is but allow bodies to be
  resolved for any accessible channel, gated on a per-familiar
  `[discord] cross_channel_peek = true` flag and optionally a size
  cap (no more than N bytes of pulled body per turn, no more than M
  fetches per minute).

## Open questions

- **Consent drift.** The operator consented to "the familiar listens
  in these channels." A user in a subscribed channel pasting a link
  to an unsubscribed channel has not consented for the familiar to
  read the target channel. How does this square with how operators
  think about the subscription boundary?
- **Spam surface.** Any user can paste any link. Unbounded body
  fetching from arbitrary links is both a rate-limit problem and an
  attack surface (prompt injection via messages sitting in
  adversarial channels).
- **Cache coherence.** A fetched body is a point-in-time snapshot.
  Should the familiar re-fetch on re-reference, or rely on history?
  If bodies are pulled into history at all, the subscribed/accessible
  boundary starts leaking retrospectively.
- **Intent signal.** Is "the user pasted a link" a strong enough
  signal to follow it? Or do we require the familiar to ask ("want
  me to look that up?") before fetching?

## Non-goals

- Crawling beyond a single linked message — no thread traversal, no
  channel scan.
- Giving the familiar standing read access to channels it isn't
  subscribed to. This page is about *ad-hoc link resolution*, not
  *passive listening*.
- Bridging the same mechanism to non-Discord platforms (Twitch, etc.).
  Cross-surface references use different plumbing.
