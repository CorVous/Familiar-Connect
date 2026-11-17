//! Backpressure policy set.

use std::collections::HashSet;

use familiar_connect::bus::protocols::BackpressurePolicy;

#[test]
fn all_policies_listed() {
    // Exactly these four policies exist.
    assert_eq!(BackpressurePolicy::ALL.len(), 4);
    let names: HashSet<&str> = BackpressurePolicy::ALL.iter().map(|p| p.as_str()).collect();
    assert_eq!(
        names,
        HashSet::from(["block", "drop_oldest", "drop_newest", "unbounded"])
    );
}
