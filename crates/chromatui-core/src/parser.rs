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

impl Default for AnsiParser {
    fn default() -> Self {
        Self::new()
    }
}
