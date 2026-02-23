use chromatui_algorithms::visual_fx;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualFxMode {
    Metaballs,
    Plasma,
    Mandelbrot,
}

pub struct VisualFxScreen {
    mode: VisualFxMode,
}

impl VisualFxScreen {
    pub fn new(mode: VisualFxMode) -> Self {
        Self { mode }
    }

    pub fn set_mode(&mut self, mode: VisualFxMode) {
        self.mode = mode;
    }

    pub fn render_ascii(&self, width: usize, height: usize, t: f64) -> Vec<String> {
        let palette: &[u8] = b" .:-=+*#%@";
        let mut lines = Vec::with_capacity(height);

        for y in 0..height {
            let mut line = String::with_capacity(width);
            for x in 0..width {
                let fx = x as f64 / width.max(1) as f64;
                let fy = y as f64 / height.max(1) as f64;
                let v = match self.mode {
                    VisualFxMode::Metaballs => {
                        let balls = [
                            (0.35 + 0.1 * t.cos(), 0.45 + 0.1 * t.sin(), 0.18),
                            (
                                0.65 + 0.1 * (t * 0.7).sin(),
                                0.55 + 0.1 * (t * 0.9).cos(),
                                0.16,
                            ),
                        ];
                        (visual_fx::metaballs_field(fx, fy, &balls) / 8.0).clamp(0.0, 1.0)
                    }
                    VisualFxMode::Plasma => {
                        let phases = [
                            (12.0, 0.0, 1.0),
                            (0.0, 11.0, 1.3),
                            (7.0, 5.0, 0.7),
                            (3.0, 13.0, 0.4),
                        ];
                        ((visual_fx::plasma_value(fx, fy, t, &phases) + 1.0) * 0.5).clamp(0.0, 1.0)
                    }
                    VisualFxMode::Mandelbrot => {
                        let cx = fx * 3.5 - 2.5;
                        let cy = fy * 2.0 - 1.0;
                        let it = visual_fx::mandelbrot_iter(cx, cy, 64) as f64;
                        (it / 64.0).clamp(0.0, 1.0)
                    }
                };

                let idx = (v * ((palette.len() - 1) as f64)).round() as usize;
                line.push(palette[idx] as char);
            }
            lines.push(line);
        }

        lines
    }
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

    #[test]
    fn visual_fx_screen_is_deterministic() {
        let fx = VisualFxScreen::new(VisualFxMode::Plasma);
        let a = fx.render_ascii(20, 8, 1.23);
        let b = fx.render_ascii(20, 8, 1.23);
        assert_eq!(a, b);
    }

    #[test]
    fn visual_fx_screen_renders_non_empty_output() {
        let fx = VisualFxScreen::new(VisualFxMode::Metaballs);
        let out = fx.render_ascii(16, 6, 0.5);
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|line| line.len() == 16));
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
