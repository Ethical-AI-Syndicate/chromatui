use std::collections::HashMap;
use taffy::prelude::*;

pub use taffy::NodeId;

#[derive(Debug, Clone)]
pub struct LayoutRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl From<Layout> for LayoutRect {
    fn from(l: Layout) -> Self {
        LayoutRect {
            x: l.location.x as i32,
            y: l.location.y as i32,
            width: l.size.width as u32,
            height: l.size.height as u32,
        }
    }
}

pub struct FlexLayout {
    tree: TaffyTree<()>,
    nodes: HashMap<NodeId, NodeId>,
    root: Option<NodeId>,
    direction: FlexDirection,
    justify_content: JustifyContent,
    align_items: AlignItems,
    align_content: AlignContent,
    gap: f32,
}

impl FlexLayout {
    pub fn new() -> Self {
        Self {
            tree: TaffyTree::new(),
            nodes: HashMap::new(),
            root: None,
            direction: FlexDirection::Column,
            justify_content: JustifyContent::Start,
            align_items: AlignItems::Stretch,
            align_content: AlignContent::Stretch,
            gap: 0.0,
        }
    }

    pub fn direction(&mut self, direction: FlexDirection) -> &mut Self {
        self.direction = direction;
        self
    }

    pub fn direction_row(&mut self) -> &mut Self {
        self.direction(FlexDirection::Row)
    }

    pub fn direction_column(&mut self) -> &mut Self {
        self.direction(FlexDirection::Column)
    }

    pub fn justify_content(&mut self, justify: JustifyContent) -> &mut Self {
        self.justify_content = justify;
        self
    }

    pub fn align_items(&mut self, align: AlignItems) -> &mut Self {
        self.align_items = align;
        self
    }

    pub fn align_content(&mut self, align: AlignContent) -> &mut Self {
        self.align_content = align;
        self
    }

    pub fn gap(&mut self, gap: f32) -> &mut Self {
        self.gap = gap;
        self
    }

    pub fn add_child(&mut self, grow: f32, shrink: f32, basis: f32) -> NodeId {
        let style = Style {
            flex_grow: grow,
            flex_shrink: shrink,
            flex_basis: length(basis),
            ..Default::default()
        };

        let node_id = self.tree.new_leaf(style).unwrap();
        self.nodes.insert(node_id, node_id);
        node_id
    }

    pub fn add_text(&mut self, width: f32, height: f32) -> NodeId {
        let style = Style {
            size: Size {
                width: length(width),
                height: length(height),
            },
            ..Default::default()
        };

        let node_id = self.tree.new_leaf(style).unwrap();
        self.nodes.insert(node_id, node_id);
        node_id
    }

    pub fn compute(&mut self, container_size: (u32, u32)) -> Vec<LayoutRect> {
        let (width, height) = container_size;

        let style = Style {
            flex_direction: self.direction,
            justify_content: Some(self.justify_content),
            align_items: Some(self.align_items),
            align_content: Some(self.align_content),
            gap: length(self.gap),
            size: Size {
                width: length(width as f32),
                height: length(height as f32),
            },
            ..Default::default()
        };

        self.root = Some(self.tree.new_leaf(style).unwrap());

        if let Some(root) = self.root {
            let _ = self.tree.compute_layout(root, Size::MAX_CONTENT);
        }

        let mut rects = Vec::new();
        for (_node_id, node) in &self.nodes {
            if let Ok(layout) = self.tree.layout(*node) {
                rects.push(LayoutRect {
                    x: layout.location.x as i32,
                    y: layout.location.y as i32,
                    width: layout.size.width as u32,
                    height: layout.size.height as u32,
                });
            }
        }

        rects
    }

    pub fn child_size(&self, child_id: NodeId) -> Option<(u32, u32)> {
        let node = self.nodes.get(&child_id)?;
        self.tree
            .layout(*node)
            .ok()
            .map(|l| (l.size.width as u32, l.size.height as u32))
    }

    pub fn child_position(&self, child_id: NodeId) -> Option<(i32, i32)> {
        let node = self.nodes.get(&child_id)?;
        self.tree
            .layout(*node)
            .ok()
            .map(|l| (l.location.x as i32, l.location.y as i32))
    }
}

impl Default for FlexLayout {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flex_layout_new() {
        let layout = FlexLayout::new();
        assert!(layout.nodes.is_empty());
    }

    #[test]
    fn test_flex_layout_direction() {
        let mut layout = FlexLayout::new();
        layout.direction_column();
        assert_eq!(layout.direction, FlexDirection::Column);

        let mut layout = FlexLayout::new();
        layout.direction_row();
        assert_eq!(layout.direction, FlexDirection::Row);
    }

    #[test]
    fn test_flex_layout_add_child() {
        let mut layout = FlexLayout::new();
        let child1 = layout.add_child(1.0, 1.0, 0.0);
        let child2 = layout.add_child(2.0, 1.0, 0.0);

        assert_ne!(child1, child2);
    }

    #[test]
    fn test_flex_layout_compute() {
        let mut layout = FlexLayout::new();
        layout.direction_column();
        layout.add_text(100.0, 50.0);
        layout.add_text(100.0, 50.0);

        let rects = layout.compute((100, 100));
        assert_eq!(rects.len(), 2);
    }

    #[test]
    fn test_flex_layout_with_grow_factors() {
        let mut layout = FlexLayout::new();
        layout.direction_column();
        layout.add_child(1.0, 1.0, 0.0);
        layout.add_child(2.0, 1.0, 0.0);

        let rects = layout.compute((100, 100));
        assert_eq!(rects.len(), 2);
    }

    #[test]
    fn test_node_id_unique() {
        let mut layout = FlexLayout::new();
        let id1 = layout.add_child(1.0, 1.0, 0.0);
        let id2 = layout.add_child(1.0, 1.0, 0.0);

        assert_ne!(id1, id2);
    }
}
