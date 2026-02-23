use crate::types::{Event, Key, MouseButton, MouseEvent, MouseEventType};
use chromatui_algorithms::capability::CapabilityPosterior;
use chromatui_algorithms::luminance::is_dark_background;
use std::io::{stdout, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

static WRITER_CLAIMED: AtomicBool = AtomicBool::new(false);

pub trait TerminalBackend {
    fn read_event(&mut self) -> Option<Event>;
    fn write(&mut self, data: &[u8]);
    fn get_size(&self) -> (u16, u16);
    fn set_raw_mode(&mut self, enabled: bool) -> std::io::Result<()>;
}

pub struct StdoutBackend {
    size: (u16, u16),
}

impl StdoutBackend {
    pub fn new() -> Self {
        Self { size: (80, 24) }
    }
}

impl Default for StdoutBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl TerminalBackend for StdoutBackend {
    fn read_event(&mut self) -> Option<Event> {
        match crossterm::event::read().ok()? {
            crossterm::event::Event::Key(key) => key_to_event_backend(key),
            crossterm::event::Event::Mouse(mouse) => Some(mouse_to_event_backend(mouse)),
            crossterm::event::Event::Resize(cols, rows) => Some(Event::Resize(cols, rows)),
            _ => None,
        }
    }

    fn write(&mut self, data: &[u8]) {
        let _ = stdout().write_all(data);
    }

    fn get_size(&self) -> (u16, u16) {
        crossterm::terminal::size().unwrap_or(self.size)
    }

    fn set_raw_mode(&mut self, enabled: bool) -> std::io::Result<()> {
        if enabled {
            crossterm::terminal::enable_raw_mode()
        } else {
            crossterm::terminal::disable_raw_mode()
        }
    }
}

pub struct TerminalSession {
    running: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct CapabilityProbeReport {
    pub alt_screen_probability: f64,
    pub truecolor_probability: f64,
    pub dark_background_probability: f64,
    pub prefers_dark: bool,
}

impl TerminalSession {
    pub fn new() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::SeqCst);
    }

    pub fn start(&self) -> std::io::Result<()> {
        use crossterm::cursor::Hide;
        use crossterm::terminal::{enable_raw_mode, EnterAlternateScreen};

        if self.is_running() {
            return Ok(());
        }

        enable_raw_mode()?;
        if let Err(err) = crossterm::execute!(stdout(), EnterAlternateScreen {}, Hide) {
            let _ = crossterm::terminal::disable_raw_mode();
            return Err(err);
        }

        self.set_running(true);
        Ok(())
    }

    pub fn stop(&self) -> std::io::Result<()> {
        use crossterm::cursor::Show;
        use crossterm::terminal::{disable_raw_mode, LeaveAlternateScreen};

        if !self.is_running() {
            return Ok(());
        }

        self.set_running(false);

        crossterm::execute!(stdout(), Show, LeaveAlternateScreen {})?;
        disable_raw_mode()?;

        Ok(())
    }

    pub fn poll_event(&self, timeout: std::time::Duration) -> Option<Event> {
        if crossterm::event::poll(timeout).ok()? {
            self.read_event()
        } else {
            None
        }
    }

    pub fn read_event(&self) -> Option<Event> {
        match crossterm::event::read().ok()? {
            crossterm::event::Event::Key(key) => self.key_to_event(key),
            crossterm::event::Event::Mouse(mouse) => Some(self.mouse_to_event(mouse)),
            crossterm::event::Event::Resize(cols, rows) => Some(Event::Resize(cols, rows)),
            _ => None,
        }
    }

    fn key_to_event(&self, key: crossterm::event::KeyEvent) -> Option<Event> {
        use crossterm::event::{KeyCode, KeyEventKind, KeyModifiers};

        if matches!(key.kind, KeyEventKind::Release) {
            return None;
        }

        let code = key.code;

        if key.modifiers.contains(KeyModifiers::ALT) {
            if let KeyCode::Char(c) = code {
                return Some(Event::Key(Key::Alt(c)));
            }
        }

        if key.modifiers.contains(KeyModifiers::CONTROL) {
            if let KeyCode::Char(c) = code {
                return Some(Event::Key(Key::Ctrl(c)));
            }
        }

        let mapped = match code {
            KeyCode::Char(c) => Key::Char(c),
            KeyCode::Enter => Key::Enter,
            KeyCode::Backspace => Key::Backspace,
            KeyCode::Delete => Key::Delete,
            KeyCode::Up => Key::Up,
            KeyCode::Down => Key::Down,
            KeyCode::Left => Key::Left,
            KeyCode::Right => Key::Right,
            KeyCode::Home => Key::Home,
            KeyCode::End => Key::End,
            KeyCode::PageUp => Key::PageUp,
            KeyCode::PageDown => Key::PageDown,
            KeyCode::Tab | KeyCode::BackTab => Key::Tab,
            KeyCode::Esc => Key::Escape,
            KeyCode::F(n) => Key::F(n),
            _ => return None,
        };

        Some(Event::Key(mapped))
    }

    fn mouse_to_event(&self, mouse: crossterm::event::MouseEvent) -> Event {
        use crossterm::event::MouseEventKind;

        let button = match mouse.kind {
            MouseEventKind::Down(b) | MouseEventKind::Up(b) | MouseEventKind::Drag(b) => {
                map_mouse_button(b)
            }
            MouseEventKind::ScrollUp => MouseButton::ScrollUp,
            MouseEventKind::ScrollDown => MouseButton::ScrollDown,
            MouseEventKind::Moved => MouseButton::Left,
            _ => MouseButton::Left,
        };

        let event_type = match mouse.kind {
            MouseEventKind::Down(_) => MouseEventType::Press,
            MouseEventKind::Up(_) => MouseEventType::Release,
            MouseEventKind::Drag(_) => MouseEventType::Drag,
            MouseEventKind::Moved | MouseEventKind::ScrollUp | MouseEventKind::ScrollDown => {
                MouseEventType::Hover
            }
            _ => MouseEventType::Hover,
        };

        Event::Mouse(MouseEvent {
            event_type,
            button,
            row: mouse.row,
            col: mouse.column,
        })
    }

    pub fn get_size(&self) -> (u16, u16) {
        crossterm::terminal::size().unwrap_or((80, 24))
    }

    pub fn probe_capabilities(&self) -> CapabilityProbeReport {
        let mut alt = CapabilityPosterior::new(0.7);
        let mut truecolor = CapabilityPosterior::new(0.6);
        let mut dark_bg = CapabilityPosterior::new(0.5);

        let term = std::env::var("TERM").unwrap_or_default().to_lowercase();
        if term.contains("xterm") || term.contains("screen") || term.contains("tmux") {
            alt.add_log_bf("term-family", 0.8);
        }
        if term.contains("256color") {
            truecolor.add_log_bf("term-256", 0.4);
        }

        let colorterm = std::env::var("COLORTERM")
            .unwrap_or_default()
            .to_lowercase();
        if colorterm.contains("truecolor") || colorterm.contains("24bit") {
            truecolor.add_log_bf("colorterm", 1.2);
        }

        if let Ok(bg) = std::env::var("COLORFGBG") {
            if let Some(last) = bg.split(';').next_back() {
                if let Ok(bg_idx) = last.parse::<u8>() {
                    let (r, g, b) = ansi_index_to_rgb(bg_idx);
                    if is_dark_background(r, g, b) {
                        dark_bg.add_log_bf("colorfgbg", 1.0);
                    } else {
                        dark_bg.add_log_bf("colorfgbg", -1.0);
                    }
                }
            }
        }

        let (w, h) = self.get_size();
        if w >= 80 && h >= 24 {
            alt.add_log_bf("size", 0.2);
        }

        let dark_p = dark_bg.probability();
        CapabilityProbeReport {
            alt_screen_probability: alt.probability(),
            truecolor_probability: truecolor.probability(),
            dark_background_probability: dark_p,
            prefers_dark: dark_p >= 0.5,
        }
    }
}

fn map_mouse_button(button: crossterm::event::MouseButton) -> MouseButton {
    match button {
        crossterm::event::MouseButton::Left => MouseButton::Left,
        crossterm::event::MouseButton::Middle => MouseButton::Middle,
        crossterm::event::MouseButton::Right => MouseButton::Right,
    }
}

fn key_to_event_backend(key: crossterm::event::KeyEvent) -> Option<Event> {
    let session = TerminalSession::new();
    session.key_to_event(key)
}

fn mouse_to_event_backend(mouse: crossterm::event::MouseEvent) -> Event {
    let session = TerminalSession::new();
    session.mouse_to_event(mouse)
}

fn ansi_index_to_rgb(idx: u8) -> (u8, u8, u8) {
    match idx {
        0 => (0, 0, 0),
        1 => (205, 0, 0),
        2 => (0, 205, 0),
        3 => (205, 205, 0),
        4 => (0, 0, 238),
        5 => (205, 0, 205),
        6 => (0, 205, 205),
        7 => (229, 229, 229),
        8 => (127, 127, 127),
        9 => (255, 0, 0),
        10 => (0, 255, 0),
        11 => (255, 255, 0),
        12 => (92, 92, 255),
        13 => (255, 0, 255),
        14 => (0, 255, 255),
        15 => (255, 255, 255),
        _ => {
            let v = idx.saturating_mul(10);
            (v, v, v)
        }
    }
}

impl Default for TerminalSession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod capability_tests {
    use super::*;
    use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};

    #[test]
    fn capability_probe_returns_probability_bounds() {
        let session = TerminalSession::new();
        let caps = session.probe_capabilities();
        assert!((0.0..=1.0).contains(&caps.alt_screen_probability));
        assert!((0.0..=1.0).contains(&caps.truecolor_probability));
    }

    #[test]
    fn unknown_key_code_is_not_mapped_to_escape() {
        let session = TerminalSession::new();
        let key = KeyEvent {
            code: KeyCode::Null,
            modifiers: KeyModifiers::empty(),
            kind: KeyEventKind::Press,
            state: KeyEventState::empty(),
        };

        assert!(session.key_to_event(key).is_none());
    }

    #[test]
    fn scroll_events_map_to_scroll_buttons() {
        let session = TerminalSession::new();
        let scroll_up = crossterm::event::MouseEvent {
            kind: crossterm::event::MouseEventKind::ScrollUp,
            column: 10,
            row: 4,
            modifiers: KeyModifiers::empty(),
        };

        let event = session.mouse_to_event(scroll_up);
        assert_eq!(
            event,
            Event::Mouse(MouseEvent {
                event_type: MouseEventType::Hover,
                button: MouseButton::ScrollUp,
                row: 4,
                col: 10,
            })
        );
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        if self.is_running() {
            let _ = self.stop();
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Inline,
    AltScreen,
}

pub struct TerminalWriter<W: Write> {
    writer: Option<W>,
    mode: OutputMode,
    owns_slot: bool,
}

impl<W: Write> TerminalWriter<W> {
    pub fn try_new(writer: W, mode: OutputMode) -> std::io::Result<Self> {
        if WRITER_CLAIMED
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                "terminal writer already active",
            ));
        }

        Ok(Self {
            writer: Some(writer),
            mode,
            owns_slot: true,
        })
    }

    pub fn mode(&self) -> OutputMode {
        self.mode
    }

    pub fn write_frame(&mut self, ansi_bytes: &[u8]) -> std::io::Result<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| std::io::Error::other("terminal writer moved"))?;

        match self.mode {
            OutputMode::Inline => {
                writer.write_all(b"\r")?;
                writer.write_all(b"\x1b[?2026h")?;
                writer.write_all(ansi_bytes)?;
                writer.write_all(b"\x1b[?2026l")?;
                writer.write_all(b"\n")?;
            }
            OutputMode::AltScreen => {
                writer.write_all(b"\x1b[?2026h")?;
                writer.write_all(ansi_bytes)?;
                writer.write_all(b"\x1b[?2026l")?;
            }
        }
        writer.flush()
    }

    pub fn into_inner(mut self) -> W {
        self.owns_slot = false;
        WRITER_CLAIMED.store(false, Ordering::SeqCst);
        self.writer
            .take()
            .expect("terminal writer should contain inner writer")
    }
}

impl TerminalWriter<std::io::Stdout> {
    pub fn try_stdout(mode: OutputMode) -> std::io::Result<Self> {
        Self::try_new(stdout(), mode)
    }
}

impl<W: Write> Drop for TerminalWriter<W> {
    fn drop(&mut self) {
        if self.owns_slot {
            WRITER_CLAIMED.store(false, Ordering::SeqCst);
        }
    }
}

#[cfg(test)]
mod writer_tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn writer_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn one_writer_rule_is_enforced() {
        let _guard = writer_test_lock()
            .lock()
            .expect("writer test lock should not be poisoned");
        let writer1 = TerminalWriter::try_new(Vec::<u8>::new(), OutputMode::Inline)
            .expect("first writer should acquire slot");
        let writer2 = TerminalWriter::try_new(Vec::<u8>::new(), OutputMode::Inline);
        assert!(writer2.is_err());
        drop(writer1);

        let writer3 = TerminalWriter::try_new(Vec::<u8>::new(), OutputMode::Inline);
        assert!(writer3.is_ok());
    }

    #[test]
    fn inline_mode_wraps_with_carriage_return_and_newline() {
        let _guard = writer_test_lock()
            .lock()
            .expect("writer test lock should not be poisoned");
        let mut writer = TerminalWriter::try_new(Vec::<u8>::new(), OutputMode::Inline)
            .expect("writer should be created");
        writer
            .write_frame(b"abc")
            .expect("write_frame should succeed");
        let bytes = writer.into_inner();
        assert_eq!(bytes, b"\r\x1b[?2026habc\x1b[?2026l\n");
    }

    #[test]
    fn alt_screen_mode_writes_raw_ansi() {
        let _guard = writer_test_lock()
            .lock()
            .expect("writer test lock should not be poisoned");
        let mut writer = TerminalWriter::try_new(Vec::<u8>::new(), OutputMode::AltScreen)
            .expect("writer should be created");
        writer
            .write_frame(b"\x1b[1;1Hok")
            .expect("write_frame should succeed");
        let bytes = writer.into_inner();
        assert_eq!(bytes, b"\x1b[?2026h\x1b[1;1Hok\x1b[?2026l");
    }
}
