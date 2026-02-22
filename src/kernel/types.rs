use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    Char(char),
    Ctrl(char),
    Alt(char),
    Enter,
    Backspace,
    Delete,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    Tab,
    Escape,
    F(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
    ScrollUp,
    ScrollDown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseEventType {
    Press,
    Release,
    Drag,
    Hover,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MouseEvent {
    pub event_type: MouseEventType,
    pub button: MouseButton,
    pub row: u16,
    pub col: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Key(Key),
    Mouse(MouseEvent),
    Resize(u16, u16),
    Timer(u64),
    Tick,
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
