use chromatui::viewport::Viewport;

#[test]
fn test_viewport_new() {
    let vp = Viewport::new(10, 80, 24);
    assert_eq!(vp.height(), 24);
}

#[test]
fn test_viewport_push_line() {
    let mut vp = Viewport::new(10, 80, 24);
    vp.push_line("hello".into());
    assert!(vp.len() > 0);
}

#[test]
fn test_viewport_scroll_offset() {
    let mut vp = Viewport::new(10, 80, 24);
    for i in 0..30 {
        vp.push_line(format!("line {}", i).into());
    }
    vp.set_scroll_offset(10);
    assert_eq!(vp.scroll_offset(), 10);
}
