use crate::types::{Event, Key};

pub struct AnsiParser {}

impl AnsiParser {
    pub fn new() -> Self {
        Self {}
    }

    pub fn parse(&mut self, input: &[u8], output: &mut Vec<u8>) {
        output.extend_from_slice(input);
    }

    pub fn parse_event(&mut self, input: &[u8]) -> Option<Event> {
        if input.is_empty() {
            return None;
        }

        if let Some(event) = parse_csi_key(input) {
            return Some(Event::Key(event));
        }

        if input.len() == 2 && input[0] == 0x1b && input[1].is_ascii() && input[1] >= 32 {
            return Some(Event::Key(Key::Alt(input[1] as char)));
        }

        match input[0] {
            b'\x1b' if input.len() == 1 => Some(Event::Key(Key::Escape)),
            b'\r' | b'\n' => Some(Event::Key(Key::Enter)),
            b'\t' => Some(Event::Key(Key::Tab)),
            b'\x7f' => Some(Event::Key(Key::Backspace)),
            c if c < 32 => Some(Event::Key(Key::Ctrl((c + 96) as char))),
            c => Some(Event::Key(Key::Char(c as char))),
        }
    }
}

fn parse_csi_key(input: &[u8]) -> Option<Key> {
    match input {
        b"\x1b[A" => Some(Key::Up),
        b"\x1b[B" => Some(Key::Down),
        b"\x1b[C" => Some(Key::Right),
        b"\x1b[D" => Some(Key::Left),
        b"\x1b[H" | b"\x1b[1~" | b"\x1bOH" => Some(Key::Home),
        b"\x1b[F" | b"\x1b[4~" | b"\x1bOF" => Some(Key::End),
        b"\x1b[5~" => Some(Key::PageUp),
        b"\x1b[6~" => Some(Key::PageDown),
        b"\x1b[3~" => Some(Key::Delete),
        _ => {
            if input.starts_with(b"\x1b[") && input.ends_with(b"~") {
                let inner = &input[2..input.len() - 1];
                if let Ok(s) = std::str::from_utf8(inner) {
                    if let Ok(n) = s.parse::<u8>() {
                        if (11..=24).contains(&n) {
                            return Some(Key::F(n - 10));
                        }
                    }
                }
            }
            None
        }
    }
}

impl Default for AnsiParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_arrow_and_nav_sequences() {
        let mut p = AnsiParser::new();
        assert_eq!(p.parse_event(b"\x1b[A"), Some(Event::Key(Key::Up)));
        assert_eq!(p.parse_event(b"\x1b[B"), Some(Event::Key(Key::Down)));
        assert_eq!(p.parse_event(b"\x1b[C"), Some(Event::Key(Key::Right)));
        assert_eq!(p.parse_event(b"\x1b[D"), Some(Event::Key(Key::Left)));
        assert_eq!(p.parse_event(b"\x1b[5~"), Some(Event::Key(Key::PageUp)));
        assert_eq!(p.parse_event(b"\x1b[6~"), Some(Event::Key(Key::PageDown)));
        assert_eq!(p.parse_event(b"\x1b[3~"), Some(Event::Key(Key::Delete)));
    }

    #[test]
    fn parses_alt_char_sequence() {
        let mut p = AnsiParser::new();
        assert_eq!(p.parse_event(b"\x1bx"), Some(Event::Key(Key::Alt('x'))));
    }
}
