pub mod diff;
pub mod viewport;

pub use diff::*;
pub use viewport::*;

pub use chromatui_layout::{FlexLayout, GridLayout, LayoutRect, NodeId};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_rect_fields() {
        let rect = LayoutRect {
            x: 10,
            y: 20,
            width: 100,
            height: 50,
        };

        assert_eq!(rect.x, 10);
        assert_eq!(rect.y, 20);
        assert_eq!(rect.width, 100);
        assert_eq!(rect.height, 50);
    }

    #[test]
    fn test_viewport_new() {
        let viewport = Viewport::new(100, 80, 24);
        assert_eq!(viewport.len(), 0);
        assert_eq!(viewport.height(), 24);
    }

    #[test]
    fn test_diff_renderer_new() {
        let renderer = DiffRenderer::new(80, 24);
        assert_eq!(renderer.width(), 80);
    }
}
