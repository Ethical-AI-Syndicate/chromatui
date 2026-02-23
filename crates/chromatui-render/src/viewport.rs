use std::collections::VecDeque;

pub struct Line {
    pub text: String,
    pub timestamp: Option<std::time::Instant>,
}

impl From<&str> for Line {
    fn from(s: &str) -> Self {
        Self {
            text: s.to_string(),
            timestamp: Some(std::time::Instant::now()),
        }
    }
}

impl From<String> for Line {
    fn from(s: String) -> Self {
        Self {
            text: s,
            timestamp: Some(std::time::Instant::now()),
        }
    }
}

pub struct Viewport {
    buffer: VecDeque<Line>,
    scroll_offset: u32,
    #[allow(dead_code)]
    cursor_row: u16,
    terminal_width: u16,
    terminal_height: u16,
    max_memory_lines: usize,
}

impl Viewport {
    pub fn new(max_memory_lines: usize, terminal_width: u16, terminal_height: u16) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_memory_lines),
            scroll_offset: 0,
            cursor_row: terminal_height.saturating_sub(1),
            terminal_width,
            terminal_height,
            max_memory_lines,
        }
    }

    pub fn push_line(&mut self, line: Line) {
        if self.buffer.len() >= self.max_memory_lines {
            self.buffer.pop_front();
        }
        self.buffer.push_back(line);
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn scroll_offset(&self) -> u32 {
        self.scroll_offset
    }

    pub fn set_scroll_offset(&mut self, offset: u32) {
        self.scroll_offset = offset.min(self.buffer.len() as u32);
    }

    pub fn height(&self) -> u16 {
        self.terminal_height
    }

    pub fn get_visible(&self) -> Vec<&Line> {
        let start = self.scroll_offset as usize;
        let end = (start + self.terminal_height as usize).min(self.buffer.len());
        self.buffer
            .iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .collect()
    }

    pub fn scroll_up(&mut self, lines: u16) {
        let max_scroll = self
            .buffer
            .len()
            .saturating_sub(self.terminal_height as usize) as u32;
        self.scroll_offset = (self.scroll_offset + lines as u32).min(max_scroll);
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = self
            .buffer
            .len()
            .saturating_sub(self.terminal_height as usize) as u32;
    }

    pub fn update_terminal_size(&mut self, width: u16, height: u16) {
        self.terminal_width = width;
        self.terminal_height = height;
    }
}
