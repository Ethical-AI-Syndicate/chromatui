use std::collections::HashMap;
use taffy::geometry::Line;
use taffy::prelude::*;
use taffy::style::{GridPlacement, GridTemplateComponent};
use taffy::style_helpers::{TaffyGridLine, TaffyGridSpan};

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

pub struct GridLayout {
    tree: TaffyTree<()>,
    nodes: HashMap<NodeId, NodeId>,
    root: Option<NodeId>,
    template_rows: Vec<GridTemplateComponent<String>>,
    template_columns: Vec<GridTemplateComponent<String>>,
    row_gap: f32,
    column_gap: f32,
}

impl GridLayout {
    pub fn new() -> Self {
        Self {
            tree: TaffyTree::new(),
            nodes: HashMap::new(),
            root: None,
            template_rows: Vec::new(),
            template_columns: Vec::new(),
            row_gap: 0.0,
            column_gap: 0.0,
        }
    }

    pub fn template_rows(&mut self, rows: Vec<GridTemplateComponent<String>>) -> &mut Self {
        self.template_rows = rows;
        self
    }

    pub fn template_columns(&mut self, cols: Vec<GridTemplateComponent<String>>) -> &mut Self {
        self.template_columns = cols;
        self
    }

    pub fn row_gap(&mut self, gap: f32) -> &mut Self {
        self.row_gap = gap;
        self
    }

    pub fn column_gap(&mut self, gap: f32) -> &mut Self {
        self.column_gap = gap;
        self
    }

    pub fn row_gap_points(&mut self, points: f32) -> &mut Self {
        self.row_gap = points;
        self
    }

    pub fn column_gap_points(&mut self, points: f32) -> &mut Self {
        self.column_gap = points;
        self
    }

    pub fn add_child_at(&mut self, row: i16, col: i16, row_span: u16, col_span: u16) -> NodeId {
        let style = Style {
            grid_row: Line {
                start: GridPlacement::<String>::from_line_index(row),
                end: GridPlacement::<String>::from_span(row_span),
            },
            grid_column: Line {
                start: GridPlacement::<String>::from_line_index(col),
                end: GridPlacement::<String>::from_span(col_span),
            },
            ..Default::default()
        };

        let node_id = self.tree.new_leaf(style).unwrap();
        self.nodes.insert(node_id, node_id);
        node_id
    }

    pub fn add_child(&mut self, row: i16, col: i16) -> NodeId {
        self.add_child_at(row, col, 1, 1)
    }

    pub fn auto_fill(&mut self, min_size: f32, _max_size: Option<f32>) {
        let track = GridTemplateComponent::Single(flex(min_size));
        self.template_columns.push(track);
    }

    pub fn auto_fit(&mut self, min_size: f32, _max_size: Option<f32>) {
        self.auto_fill(min_size, None);
    }

    pub fn compute(&mut self, container_size: (u32, u32)) -> Vec<LayoutRect> {
        let (width, height) = container_size;

        let style = Style {
            grid_template_rows: self.template_rows.clone(),
            grid_template_columns: self.template_columns.clone(),
            gap: Size {
                width: LengthPercentage::length(self.column_gap),
                height: LengthPercentage::length(self.row_gap),
            },
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
        for node in self.nodes.values() {
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

impl Default for GridLayout {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_layout_new() {
        let layout = GridLayout::new();
        assert!(layout.nodes.is_empty());
    }

    #[test]
    fn test_grid_layout_add_child() {
        let mut layout = GridLayout::new();
        let child1 = layout.add_child(0, 0);
        let child2 = layout.add_child(0, 1);

        assert_ne!(child1, child2);
    }

    #[test]
    fn test_grid_layout_with_spans() {
        let mut layout = GridLayout::new();
        let child = layout.add_child_at(0, 0, 2, 2);

        assert!(layout.nodes.contains_key(&child));
    }

    #[test]
    fn test_grid_layout_compute() {
        let mut layout = GridLayout::new();
        layout.template_rows(vec![GridTemplateComponent::Single(length(50.0))]);
        layout.template_columns(vec![
            GridTemplateComponent::Single(length(50.0)),
            GridTemplateComponent::Single(length(50.0)),
        ]);

        layout.add_child(0, 0);
        layout.add_child(0, 1);

        let rects = layout.compute((100, 100));
        assert_eq!(rects.len(), 2);
    }

    #[test]
    fn test_grid_layout_with_gaps() {
        let mut layout = GridLayout::new();
        layout.row_gap_points(5.0);
        layout.column_gap_points(5.0);

        assert_eq!(layout.row_gap, 5.0);
        assert_eq!(layout.column_gap, 5.0);
    }

    #[test]
    fn test_node_id_unique_in_grid() {
        let mut layout = GridLayout::new();
        let id1 = layout.add_child(0, 0);
        let id2 = layout.add_child(1, 0);

        assert_ne!(id1, id2);
    }
}
