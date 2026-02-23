use crate::types::{Event, Key, MouseButton, MouseEvent, MouseEventType};
use std::io::{stdout, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

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

impl TerminalBackend for StdoutBackend {
    fn read_event(&mut self) -> Option<Event> {
        None
    }

    fn write(&mut self, data: &[u8]) {
        let _ = stdout().write_all(data);
    }

    fn get_size(&self) -> (u16, u16) {
        self.size
    }

    fn set_raw_mode(&mut self, _enabled: bool) -> std::io::Result<()> {
        Ok(())
    }
}

pub struct TerminalSession {
    running: Arc<AtomicBool>,
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
        use crossterm::terminal::{enable_raw_mode, EnterAlternateScreen};

        enable_raw_mode()?;
        crossterm::execute!(stdout(), EnterAlternateScreen {})?;

        self.set_running(true);
        Ok(())
    }

    pub fn stop(&self) -> std::io::Result<()> {
        use crossterm::terminal::{disable_raw_mode, LeaveAlternateScreen};

        self.set_running(false);

        crossterm::execute!(stdout(), LeaveAlternateScreen {})?;
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
            crossterm::event::Event::Key(key) => Some(self.key_to_event(key)),
            crossterm::event::Event::Mouse(mouse) => Some(self.mouse_to_event(mouse)),
            crossterm::event::Event::Resize(cols, rows) => Some(Event::Resize(cols, rows)),
            _ => None,
        }
    }

    fn key_to_event(&self, key: crossterm::event::KeyEvent) -> Event {
        let code = key.code;

        if key.modifiers.contains(crossterm::event::KeyModifiers::ALT) {
            if let crossterm::event::KeyCode::Char(c) = code {
                return Event::Key(Key::Alt(c));
            }
        }

        if key
            .modifiers
            .contains(crossterm::event::KeyModifiers::CONTROL)
        {
            if let crossterm::event::KeyCode::Char(c) = code {
                return Event::Key(Key::Ctrl(c));
            }
        }

        match code {
            crossterm::event::KeyCode::Char(c) => Event::Key(Key::Char(c)),
            crossterm::event::KeyCode::Enter => Event::Key(Key::Enter),
            crossterm::event::KeyCode::Backspace => Event::Key(Key::Backspace),
            crossterm::event::KeyCode::Delete => Event::Key(Key::Delete),
            crossterm::event::KeyCode::Up => Event::Key(Key::Up),
            crossterm::event::KeyCode::Down => Event::Key(Key::Down),
            crossterm::event::KeyCode::Left => Event::Key(Key::Left),
            crossterm::event::KeyCode::Right => Event::Key(Key::Right),
            crossterm::event::KeyCode::Home => Event::Key(Key::Home),
            crossterm::event::KeyCode::End => Event::Key(Key::End),
            crossterm::event::KeyCode::PageUp => Event::Key(Key::PageUp),
            crossterm::event::KeyCode::PageDown => Event::Key(Key::PageDown),
            crossterm::event::KeyCode::Tab => Event::Key(Key::Tab),
            crossterm::event::KeyCode::Esc => Event::Key(Key::Escape),
            crossterm::event::KeyCode::F(n) => Event::Key(Key::F(n)),
            _ => Event::Key(Key::Escape),
        }
    }

    fn mouse_to_event(&self, mouse: crossterm::event::MouseEvent) -> Event {
        let button = match mouse.kind {
            crossterm::event::MouseEventKind::Down(b) => match b {
                crossterm::event::MouseButton::Left => MouseButton::Left,
                crossterm::event::MouseButton::Middle => MouseButton::Middle,
                crossterm::event::MouseButton::Right => MouseButton::Right,
            },
            crossterm::event::MouseEventKind::Up(_) => MouseButton::Left,
            crossterm::event::MouseEventKind::Drag(_) => MouseButton::Left,
            crossterm::event::MouseEventKind::Moved => MouseButton::Left,
            _ => MouseButton::Left,
        };

        let event_type = match mouse.kind {
            crossterm::event::MouseEventKind::Down(_) => MouseEventType::Press,
            crossterm::event::MouseEventKind::Up(_) => MouseEventType::Release,
            crossterm::event::MouseEventKind::Drag(_) => MouseEventType::Drag,
            crossterm::event::MouseEventKind::Moved => MouseEventType::Hover,
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
        crossterm::terminal::size()
            .map(|(c, r)| (c, r))
            .unwrap_or((80, 24))
    }
}

impl Default for TerminalSession {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        if self.is_running() {
            let _ = self.stop();
        }
    }
}
