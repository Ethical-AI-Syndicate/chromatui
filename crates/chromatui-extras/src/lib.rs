#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyHint {
    pub key: String,
    pub description: String,
}

impl KeyHint {
    pub fn new(key: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            description: description.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusBar {
    width: usize,
    left: String,
    right: String,
}

impl StatusBar {
    pub fn new(width: usize) -> Self {
        Self {
            width,
            left: String::new(),
            right: String::new(),
        }
    }

    pub fn set_left(&mut self, value: impl Into<String>) {
        self.left = value.into();
    }

    pub fn set_right(&mut self, value: impl Into<String>) {
        self.right = value.into();
    }

    pub fn render(&self) -> String {
        if self.width == 0 {
            return String::new();
        }

        let mut line = self.left.clone();
        if line.len() >= self.width {
            line.truncate(self.width);
            return line;
        }

        let right = if self.right.len() > self.width {
            &self.right[self.right.len() - self.width..]
        } else {
            &self.right
        };

        let free = self.width.saturating_sub(line.len());
        if right.len() >= free {
            line.push_str(right);
            line.truncate(self.width);
            return line;
        }

        let padding = free - right.len();
        line.push_str(&" ".repeat(padding));
        line.push_str(right);
        line
    }
}

#[cfg(feature = "syntax-highlighting")]
pub mod syntax {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum TokenKind {
        Keyword,
        Number,
        String,
        Comment,
        Plain,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TokenSpan {
        pub start: usize,
        pub end: usize,
        pub kind: TokenKind,
    }

    const KEYWORDS: [&str; 8] = ["fn", "let", "mut", "struct", "enum", "impl", "pub", "use"];

    pub fn highlight_rust_line(line: &str) -> Vec<TokenSpan> {
        if let Some(comment_start) = line.find("//") {
            let mut spans = classify_code(&line[..comment_start]);
            spans.push(TokenSpan {
                start: comment_start,
                end: line.len(),
                kind: TokenKind::Comment,
            });
            return spans;
        }

        classify_code(line)
    }

    fn classify_code(line: &str) -> Vec<TokenSpan> {
        let mut spans = Vec::new();
        let bytes = line.as_bytes();
        let mut i = 0usize;

        while i < bytes.len() {
            let c = bytes[i] as char;

            if c.is_whitespace() {
                i += 1;
                continue;
            }

            if c == '"' {
                let start = i;
                i += 1;
                while i < bytes.len() && bytes[i] as char != '"' {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1;
                }
                spans.push(TokenSpan {
                    start,
                    end: i,
                    kind: TokenKind::String,
                });
                continue;
            }

            if c.is_ascii_digit() {
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i] as char).is_ascii_digit() {
                    i += 1;
                }
                spans.push(TokenSpan {
                    start,
                    end: i,
                    kind: TokenKind::Number,
                });
                continue;
            }

            let start = i;
            while i < bytes.len() {
                let ch = bytes[i] as char;
                if ch.is_whitespace() || ch == '"' || ch == '/' {
                    break;
                }
                i += 1;
            }

            let token = &line[start..i];
            let kind = if KEYWORDS.contains(&token) {
                TokenKind::Keyword
            } else {
                TokenKind::Plain
            };

            spans.push(TokenSpan {
                start,
                end: i,
                kind,
            });
        }

        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_bar_layout() {
        let mut bar = StatusBar::new(20);
        bar.set_left("NORMAL");
        bar.set_right("1:1");
        let rendered = bar.render();

        assert_eq!(rendered.len(), 20);
        assert!(rendered.starts_with("NORMAL"));
        assert!(rendered.ends_with("1:1"));
    }

    #[cfg(feature = "syntax-highlighting")]
    #[test]
    fn rust_highlighting_tokens() {
        let spans = syntax::highlight_rust_line("pub fn run() { let x = 42 } // comment");
        assert!(spans.iter().any(|s| s.kind == syntax::TokenKind::Keyword));
        assert!(spans.iter().any(|s| s.kind == syntax::TokenKind::Number));
        assert!(spans.iter().any(|s| s.kind == syntax::TokenKind::Comment));
    }
}
