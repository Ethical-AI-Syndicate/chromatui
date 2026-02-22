use chromatui::renderer::{DiffRenderer, Region};

#[test]
fn test_region_creation() {
    let region = Region::new(0, 0, 10, 80);
    assert!(region.is_valid());
}

#[test]
fn test_diff_renderer_new() {
    let renderer = DiffRenderer::new(80, 24);
    assert_eq!(renderer.width(), 80);
}
