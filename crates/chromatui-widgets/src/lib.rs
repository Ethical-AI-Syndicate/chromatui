use chromatui_core::Key;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InlineAction {
    Submit(String),
    Cancel(String),
    Complete(String),
}

pub struct InlineEditor {
    prompt: String,
    content: String,
    cursor_pos: usize,
    history: Vec<String>,
    history_index: Option<usize>,
}

impl InlineEditor {
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            content: String::new(),
            cursor_pos: 0,
            history: Vec::new(),
            history_index: None,
        }
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn cursor_pos(&self) -> usize {
        self.cursor_pos
    }

    pub fn handle_key(&mut self, key: Key) -> Option<InlineAction> {
        match key {
            Key::Char(c) => {
                if self.cursor_pos > self.content.len() {
                    self.cursor_pos = self.content.len();
                }
                self.content.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
                self.history_index = None;
                None
            }
            Key::Backspace => {
                if self.cursor_pos > 0 && !self.content.is_empty() {
                    self.content.remove(self.cursor_pos - 1);
                    self.cursor_pos = self.cursor_pos.saturating_sub(1);
                }
                self.history_index = None;
                None
            }
            Key::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
                None
            }
            Key::Right => {
                self.cursor_pos = (self.cursor_pos + 1).min(self.content.len());
                None
            }
            Key::Home => {
                self.cursor_pos = 0;
                None
            }
            Key::End => {
                self.cursor_pos = self.content.len();
                None
            }
            Key::Enter => {
                let submitted = self.content.clone();
                if !submitted.is_empty() {
                    self.history.push(submitted.clone());
                }
                let result = Some(InlineAction::Submit(std::mem::take(&mut self.content)));
                self.cursor_pos = 0;
                self.history_index = None;
                result
            }
            Key::Escape => {
                let cancelled = self.content.clone();
                self.content.clear();
                self.cursor_pos = 0;
                Some(InlineAction::Cancel(cancelled))
            }
            Key::Up => {
                if let Some(idx) = self.history_index {
                    if idx + 1 < self.history.len() {
                        self.history_index = Some(idx + 1);
                    }
                } else if !self.history.is_empty() {
                    self.history_index = Some(0);
                }
                if let Some(idx) = self.history_index {
                    self.content = self.history[idx].clone();
                    self.cursor_pos = self.content.len();
                }
                None
            }
            Key::Down => {
                if let Some(idx) = self.history_index {
                    if idx == 0 {
                        self.history_index = None;
                        self.content.clear();
                    } else {
                        self.history_index = Some(idx - 1);
                        self.content = self.history[idx - 1].clone();
                    }
                }
                self.cursor_pos = self.content.len();
                None
            }
            _ => None,
        }
    }

    pub fn render(&self) -> String {
        let mut output = self.prompt.clone();
        output.push_str(&self.content);
        output
    }

    pub fn submit(&self) -> String {
        self.content.clone()
    }

    pub fn cancel(&self) -> String {
        self.content.clone()
    }
}
