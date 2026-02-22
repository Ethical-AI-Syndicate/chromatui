use chromatui::runtime::{Runtime, RuntimeState};

#[test]
fn test_runtime_initial_state() {
    let runtime = Runtime::new();
    matches!(runtime.state(), RuntimeState::Normal);
}

#[test]
fn test_runtime_timers() {
    let mut runtime = Runtime::new();
    runtime.add_timer(1000, 1);
    assert!(runtime.has_timer(1));
}
