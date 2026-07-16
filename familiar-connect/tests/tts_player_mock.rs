//! Port of `tests/test_tts_player_mock.py` — the mock player honours
//! `TurnScope` cancellation and `stop`, so the barge-in integration test is
//! meaningful.

use std::sync::Arc;
use std::time::Duration;

use familiar_connect::bus::envelope::TurnScope;
use familiar_connect::tts_player::MockTTSPlayer;
use familiar_connect::tts_player::protocol::TtsPlayer;

#[tokio::test]
async fn plays_full_text_when_not_cancelled() {
    let player = MockTTSPlayer::new(5, 5);
    let scope = TurnScope::new("t", "s");
    player.speak("hello world four words", &scope).await;
    // Four words x 5ms = 20ms min; allow slack.
    assert!(player.total_played_ms() >= 15);
    assert_eq!(
        player.calls(),
        vec![("hello world four words".to_owned(), false)]
    );
}

#[tokio::test]
async fn stops_promptly_on_cancel() {
    let player = MockTTSPlayer::new(20, 5);
    let scope = Arc::new(TurnScope::new("t", "s"));
    let scope2 = Arc::clone(&scope);
    let canceller = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(30)).await;
        scope2.cancel();
    });
    let text = "one two three four five six seven eight nine ten";
    let t0 = std::time::Instant::now();
    player.speak(text, scope.as_ref()).await;
    let elapsed = t0.elapsed();
    canceller.await.unwrap();

    // Should stop within ~cancel_delay + poll_ms, not play all 200ms.
    assert!(elapsed < Duration::from_millis(80), "took {elapsed:?}");
    assert!(player.total_played_ms() < 80);
    assert_eq!(player.calls(), vec![(text.to_owned(), true)]);
}

#[tokio::test]
async fn stop_flushes_current_playback() {
    let player = Arc::new(MockTTSPlayer::new(30, 5));
    let scope = TurnScope::new("t", "s");
    let player2 = Arc::clone(&player);
    let stopper = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        player2.stop().await;
    });
    let t0 = std::time::Instant::now();
    player.speak("one two three four", &scope).await;
    let elapsed = t0.elapsed();
    stopper.await.unwrap();

    assert!(elapsed < Duration::from_millis(60), "took {elapsed:?}");
}
