#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    Plain,
    Emphasis,
    Strong,
    Code,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub text: String,
    pub kind: SegmentKind,
}

impl Segment {
    pub fn new(text: impl Into<String>, kind: SegmentKind) -> Self {
        Self {
            text: text.into(),
            kind,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TextStyle {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub style: TextStyle,
}

impl Span {
    pub fn new(start: usize, end: usize, style: TextStyle) -> Option<Self> {
        if start <= end {
            Some(Self { start, end, style })
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RopeText {
    lines: Vec<String>,
}

impl RopeText {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_text(text: &str) -> Self {
        let lines = text.split('\n').map(ToOwned::to_owned).collect();
        Self { lines }
    }

    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty() || (self.lines.len() == 1 && self.lines[0].is_empty())
    }

    pub fn line(&self, index: usize) -> Option<&str> {
        self.lines.get(index).map(String::as_str)
    }

    pub fn insert_line(&mut self, index: usize, line: impl Into<String>) {
        let insert_at = index.min(self.lines.len());
        self.lines.insert(insert_at, line.into());
    }

    pub fn remove_line(&mut self, index: usize) -> Option<String> {
        if index < self.lines.len() {
            Some(self.lines.remove(index))
        } else {
            None
        }
    }

    pub fn replace_line(&mut self, index: usize, line: impl Into<String>) -> bool {
        if let Some(slot) = self.lines.get_mut(index) {
            *slot = line.into();
            true
        } else {
            false
        }
    }

    pub fn push_line(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }

    pub fn to_text(&self) -> String {
        self.lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_validation() {
        assert!(Span::new(1, 3, TextStyle::default()).is_some());
        assert!(Span::new(3, 1, TextStyle::default()).is_none());
    }

    #[test]
    fn rope_round_trip() {
        let rope = RopeText::from_text("alpha\nbeta\ngamma");
        assert_eq!(rope.line_count(), 3);
        assert_eq!(rope.line(1), Some("beta"));
        assert_eq!(rope.to_text(), "alpha\nbeta\ngamma");
    }

    #[test]
    fn rope_line_operations() {
        let mut rope = RopeText::new();
        rope.push_line("a");
        rope.push_line("c");
        rope.insert_line(1, "b");

        assert_eq!(rope.to_text(), "a\nb\nc");
        assert!(rope.replace_line(2, "z"));
        assert_eq!(rope.remove_line(0), Some(String::from("a")));
        assert_eq!(rope.to_text(), "b\nz");
    }
}
