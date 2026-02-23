use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return None;
        }
        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
        Some(Color { r, g, b })
    }

    pub fn to_truecolor(&self) -> String {
        format!("\x1b[38;2;{};{};{}m", self.r, self.g, self.b)
    }

    pub fn to_ansi256(&self) -> u8 {
        if self.r == self.g && self.g == self.b {
            if self.r < 8 {
                return 16;
            }
            if self.r > 248 {
                return 231;
            }
            let gray = (self.r - 8) / 10 + 232;
            return gray;
        }

        let r = self.r / 51;
        let g = self.g / 51;
        let b = self.b / 51;

        16 + r * 36 + g * 6 + b
    }

    pub fn to_ansi256_escape(&self) -> String {
        format!("\x1b[38;5;{}m", self.to_ansi256())
    }

    pub fn terminal_default() -> Self {
        Self {
            r: 255,
            g: 255,
            b: 255,
        }
    }

    pub fn black() -> Self {
        Self { r: 0, g: 0, b: 0 }
    }
    pub fn white() -> Self {
        Self {
            r: 255,
            g: 255,
            b: 255,
        }
    }
    pub fn red() -> Self {
        Self { r: 255, g: 0, b: 0 }
    }
    pub fn green() -> Self {
        Self { r: 0, g: 255, b: 0 }
    }
    pub fn blue() -> Self {
        Self { r: 0, g: 0, b: 255 }
    }
    pub fn yellow() -> Self {
        Self {
            r: 255,
            g: 255,
            b: 0,
        }
    }
    pub fn cyan() -> Self {
        Self {
            r: 0,
            g: 255,
            b: 255,
        }
    }
    pub fn magenta() -> Self {
        Self {
            r: 255,
            g: 0,
            b: 255,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hex_with_hash() {
        let color = Color::from_hex("#FF5500").unwrap();
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 85);
        assert_eq!(color.b, 0);
    }

    #[test]
    fn test_color_from_hex_without_hash() {
        let color = Color::from_hex("FF5500").unwrap();
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 85);
        assert_eq!(color.b, 0);
    }

    #[test]
    fn test_color_from_hex_invalid() {
        assert!(Color::from_hex("invalid").is_none());
        assert!(Color::from_hex("#GG5500").is_none());
        assert!(Color::from_hex("#FF55").is_none());
    }

    #[test]
    fn test_color_to_truecolor() {
        let color = Color::rgb(255, 85, 0);
        assert_eq!(color.to_truecolor(), "\x1b[38;2;255;85;0m");
    }

    #[test]
    fn test_color_to_ansi256() {
        let color = Color::rgb(0, 0, 0);
        assert_eq!(color.to_ansi256(), 16);

        let white = Color::rgb(255, 255, 255);
        assert_eq!(white.to_ansi256(), 231);
    }

    #[test]
    fn test_color_rgb_constructor() {
        let color = Color::rgb(100, 150, 200);
        assert_eq!(color.r, 100);
        assert_eq!(color.g, 150);
        assert_eq!(color.b, 200);
    }
}
