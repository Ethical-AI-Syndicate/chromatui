use chromatui_style::Style;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WidgetId(pub usize);

impl WidgetId {
    pub fn new(id: usize) -> Self {
        WidgetId(id)
    }
}

#[derive(Debug, Clone)]
pub enum Node {
    Leaf(LeafNode),
    Element(ElementNode),
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Node::Leaf(a), Node::Leaf(b)) => a.id == b.id && a.content == b.content,
            (Node::Element(a), Node::Element(b)) => {
                a.id == b.id && a.tag == b.tag && a.children == b.children
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LeafNode {
    pub id: WidgetId,
    pub content: String,
    pub style: Option<Style>,
}

impl PartialEq for LeafNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.content == other.content
    }
}

#[derive(Debug, Clone)]
pub struct ElementNode {
    pub id: WidgetId,
    pub tag: String,
    pub children: Vec<Node>,
    pub style: Option<Style>,
    pub layout: Option<LayoutRef>,
}

impl PartialEq for ElementNode {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.tag == other.tag && self.children == other.children
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LayoutRef {
    Flex(FlexConfig),
    Grid(GridConfig),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FlexConfig {
    pub direction: FlexDirection,
    pub justify_content: Option<JustifyContent>,
    pub align_items: Option<AlignItems>,
    pub gap: f32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum FlexDirection {
    #[default]
    Row,
    Column,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JustifyContent {
    Start,
    End,
    Center,
    SpaceBetween,
    SpaceAround,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlignItems {
    Stretch,
    Start,
    End,
    Center,
}

impl Default for AlignItems {
    fn default() -> Self {
        AlignItems::Stretch
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GridConfig {
    pub rows: Vec<TrackSize>,
    pub columns: Vec<TrackSize>,
    pub row_gap: f32,
    pub column_gap: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrackSize {
    Auto,
    Fixed(f32),
    Fr(f32),
    MinContent,
    MaxContent,
}

pub struct View {
    pub root: Node,
}

impl View {
    pub fn new(root: Node) -> Self {
        View { root }
    }

    pub fn text(content: &str) -> Node {
        Node::Leaf(LeafNode {
            id: WidgetId::new(0),
            content: content.to_string(),
            style: None,
        })
    }

    pub fn element(tag: &str, children: Vec<Node>) -> Node {
        Node::Element(ElementNode {
            id: WidgetId::new(0),
            tag: tag.to_string(),
            children,
            style: None,
            layout: None,
        })
    }

    pub fn container(children: Vec<Node>) -> Node {
        Self::element("container", children)
    }

    pub fn row(children: Vec<Node>) -> Node {
        let mut node = Self::element("row", children);
        if let Node::Element(ref mut elem) = node {
            elem.layout = Some(LayoutRef::Flex(FlexConfig {
                direction: FlexDirection::Row,
                ..Default::default()
            }));
        }
        node
    }

    pub fn column(children: Vec<Node>) -> Node {
        let mut node = Self::element("column", children);
        if let Node::Element(ref mut elem) = node {
            elem.layout = Some(LayoutRef::Flex(FlexConfig {
                direction: FlexDirection::Column,
                ..Default::default()
            }));
        }
        node
    }
}

#[derive(Debug, Clone)]
pub struct Signal<T: Clone> {
    value: T,
    subscribers: Vec<SubscriberId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriberId(usize);

#[derive(Debug, Default)]
pub struct State {
    signals: std::collections::HashMap<usize, Box<dyn std::any::Any>>,
    next_subscriber_id: usize,
}

impl State {
    pub fn new() -> Self {
        State {
            signals: std::collections::HashMap::new(),
            next_subscriber_id: 0,
        }
    }

    pub fn signal<T: Clone + 'static>(&mut self, value: T) -> Signal<T> {
        let id = self.next_subscriber_id;
        self.next_subscriber_id += 1;
        self.signals.insert(id, Box::new(value.clone()));
        Signal {
            value,
            subscribers: vec![SubscriberId(id)],
        }
    }

    pub fn get<T: Clone + 'static>(&self, signal: &Signal<T>) -> T {
        signal.value.clone()
    }

    pub fn set<T: Clone + 'static>(&mut self, signal: &mut Signal<T>, value: T) {
        signal.value = value.clone();
        if let Some(stored) = self.signals.get(&signal.subscribers[0].0) {
            if stored.downcast_ref::<T>().is_some() {
                let _ = self
                    .signals
                    .insert(signal.subscribers[0].0, Box::new(value));
            }
        }
    }
}

impl<T: Clone> Signal<T> {
    pub fn new(value: T) -> Self {
        Signal {
            value,
            subscribers: Vec::new(),
        }
    }

    pub fn get(&self) -> T {
        self.value.clone()
    }

    pub fn set(&mut self, value: T) {
        self.value = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_widget_id() {
        let id = WidgetId::new(42);
        assert_eq!(id.0, 42);
    }

    #[test]
    fn test_view_text() {
        let node = View::text("Hello, World!");
        match node {
            Node::Leaf(leaf) => {
                assert_eq!(leaf.content, "Hello, World!");
            }
            _ => panic!("Expected Leaf node"),
        }
    }

    #[test]
    fn test_view_element() {
        let children = vec![View::text("child1"), View::text("child2")];
        let node = View::element("div", children);
        match node {
            Node::Element(elem) => {
                assert_eq!(elem.tag, "div");
                assert_eq!(elem.children.len(), 2);
            }
            _ => panic!("Expected Element node"),
        }
    }

    #[test]
    fn test_view_row() {
        let children = vec![View::text("a"), View::text("b")];
        let node = View::row(children);
        match node {
            Node::Element(elem) => {
                if let Some(LayoutRef::Flex(config)) = &elem.layout {
                    assert_eq!(config.direction, FlexDirection::Row);
                } else {
                    panic!("Expected Flex layout");
                }
            }
            _ => panic!("Expected Element node"),
        }
    }

    #[test]
    fn test_view_column() {
        let children = vec![View::text("a"), View::text("b")];
        let node = View::column(children);
        match node {
            Node::Element(elem) => {
                if let Some(LayoutRef::Flex(config)) = &elem.layout {
                    assert_eq!(config.direction, FlexDirection::Column);
                } else {
                    panic!("Expected Flex layout");
                }
            }
            _ => panic!("Expected Element node"),
        }
    }

    #[test]
    fn test_signal_new() {
        let signal = Signal::new(42);
        assert_eq!(signal.get(), 42);
    }

    #[test]
    fn test_signal_set() {
        let mut signal = Signal::new(10);
        signal.set(20);
        assert_eq!(signal.get(), 20);
    }

    #[test]
    fn test_state_signal() {
        let mut state = State::new();
        let signal = state.signal(100);
        assert_eq!(state.get(&signal), 100);
    }

    #[test]
    fn test_state_signal_update() {
        let mut state = State::new();
        let mut signal = state.signal(100);
        state.set(&mut signal, 200);
        assert_eq!(state.get(&signal), 200);
    }
}
