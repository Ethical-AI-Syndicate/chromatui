use super::Color;

#[derive(Debug, Clone)]
pub struct Theme {
    pub name: String,
    pub colors: ColorScheme,
    pub styles: StyleMap,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            name: "default".into(),
            colors: ColorScheme::default(),
            styles: StyleMap::default(),
        }
    }
}

impl Theme {
    pub fn dark() -> Self {
        Self {
            name: "dark".into(),
            colors: ColorScheme::dark(),
            styles: StyleMap::default(),
        }
    }

    pub fn light() -> Self {
        Self {
            name: "light".into(),
            colors: ColorScheme::light(),
            styles: StyleMap::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ColorScheme {
    pub foreground: Color,
    pub background: Color,
    pub primary: Color,
    pub secondary: Color,
    pub accent: Color,
    pub error: Color,
    pub warning: Color,
    pub success: Color,
    pub dim: Color,
    pub bright: Color,
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self::dark()
    }
}

impl ColorScheme {
    pub fn dark() -> Self {
        Self {
            foreground: Color::rgb(255, 255, 255),
            background: Color::rgb(0, 0, 0),
            primary: Color::rgb(0, 122, 204),
            secondary: Color::rgb(128, 128, 128),
            accent: Color::rgb(0, 255, 255),
            error: Color::rgb(255, 85, 85),
            warning: Color::rgb(255, 170, 0),
            success: Color::rgb(85, 255, 85),
            dim: Color::rgb(128, 128, 128),
            bright: Color::rgb(255, 255, 255),
        }
    }

    pub fn light() -> Self {
        Self {
            foreground: Color::rgb(0, 0, 0),
            background: Color::rgb(255, 255, 255),
            primary: Color::rgb(0, 100, 180),
            secondary: Color::rgb(128, 128, 128),
            accent: Color::rgb(0, 150, 200),
            error: Color::rgb(200, 0, 0),
            warning: Color::rgb(200, 130, 0),
            success: Color::rgb(0, 150, 0),
            dim: Color::rgb(100, 100, 100),
            bright: Color::rgb(0, 0, 0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StyleMap {
    pub button: Style,
    pub input: Style,
    pub text: Style,
    pub border: Style,
}

impl Default for StyleMap {
    fn default() -> Self {
        Self {
            button: Style::default_button(),
            input: Style::default_input(),
            text: Style::default_text(),
            border: Style::default_border(),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Style {
    pub fg: Option<Color>,
    pub bg: Option<Color>,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub strikethrough: bool,
}

impl Style {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fg(mut self, color: Color) -> Self {
        self.fg = Some(color);
        self
    }

    pub fn bg(mut self, color: Color) -> Self {
        self.bg = Some(color);
        self
    }

    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }

    pub fn italic(mut self) -> Self {
        self.italic = true;
        self
    }

    pub fn underline(mut self) -> Self {
        self.underline = true;
        self
    }

    pub fn strikethrough(mut self) -> Self {
        self.strikethrough = true;
        self
    }

    pub fn to_escape_sequence(&self) -> String {
        let mut seq = String::new();

        if let Some(fg) = &self.fg {
            seq.push_str(&fg.to_truecolor());
        }

        if let Some(bg) = &self.bg {
            seq.push_str(&format!("\x1b[48;2;{};{};{}m", bg.r, bg.g, bg.b));
        }

        if self.bold {
            seq.push_str("\x1b[1m");
        }
        if self.italic {
            seq.push_str("\x1b[3m");
        }
        if self.underline {
            seq.push_str("\x1b[4m");
        }
        if self.strikethrough {
            seq.push_str("\x1b[9m");
        }

        seq
    }

    pub fn default_button() -> Self {
        Self {
            fg: Some(Color::rgb(255, 255, 255)),
            bg: Some(Color::rgb(0, 122, 204)),
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
        }
    }

    pub fn default_input() -> Self {
        Self {
            fg: Some(Color::rgb(255, 255, 255)),
            bg: Some(Color::rgb(30, 30, 30)),
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
        }
    }

    pub fn default_text() -> Self {
        Self {
            fg: Some(Color::rgb(255, 255, 255)),
            bg: None,
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
        }
    }

    pub fn default_border() -> Self {
        Self {
            fg: Some(Color::rgb(128, 128, 128)),
            bg: None,
            bold: false,
            italic: false,
            underline: false,
            strikethrough: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theme_default() {
        let theme = Theme::default();
        assert!(!theme.name.is_empty());
        assert!(
            theme.colors.foreground.r > 0
                || theme.colors.foreground.g > 0
                || theme.colors.foreground.b > 0
        );
    }

    #[test]
    fn test_color_scheme_dark() {
        let colors = ColorScheme::dark();
        assert_eq!(colors.background, Color::rgb(0, 0, 0));
        assert_eq!(colors.foreground, Color::rgb(255, 255, 255));
    }

    #[test]
    fn test_color_scheme_light() {
        let colors = ColorScheme::light();
        assert_eq!(colors.background, Color::rgb(255, 255, 255));
        assert_eq!(colors.foreground, Color::rgb(0, 0, 0));
    }

    #[test]
    fn test_style_builder() {
        let style = Style::new()
            .fg(Color::red())
            .bg(Color::black())
            .bold()
            .italic();

        assert!(style.fg.is_some());
        assert!(style.bg.is_some());
        assert!(style.bold);
        assert!(style.italic);
    }

    #[test]
    fn test_style_to_escape() {
        let style = Style::new().fg(Color::rgb(255, 0, 0)).bold();

        let escape = style.to_escape_sequence();
        assert!(escape.contains("\x1b[38;2;255;0;0m"));
        assert!(escape.contains("\x1b[1m"));
    }
}
