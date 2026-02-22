#[test]
fn test_key_equality() {
    use chromatui::kernel::types::Key;
    assert_eq!(Key::Char('a'), Key::Char('a'));
    assert_ne!(Key::Char('a'), Key::Char('b'));
}

#[test]
fn test_event_display() {
    use chromatui::kernel::types::Event;
    let event = Event::Key(chromatui::kernel::types::Key::Char('x'));
    let _ = format!("{:?}", event);
}
