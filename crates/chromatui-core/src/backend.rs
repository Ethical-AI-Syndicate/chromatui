use crate::types::Event;
use std::io::Write;

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
        let _ = std::io::stdout().write_all(data);
    }

    fn get_size(&self) -> (u16, u16) {
        self.size
    }

    fn set_raw_mode(&mut self, _enabled: bool) -> std::io::Result<()> {
        Ok(())
    }
}
