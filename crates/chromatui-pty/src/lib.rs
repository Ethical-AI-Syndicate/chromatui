use portable_pty::{native_pty_system, CommandBuilder, PtyPair, PtySize};
use std::io::{Read, Write};
use std::thread;

pub struct PtySession {
    pair: PtyPair,
    reader: Option<Box<dyn Read + Send>>,
}

impl PtySession {
    pub fn new(rows: u16, cols: u16) -> Result<Self, Box<dyn std::error::Error>> {
        let pty_system = native_pty_system();
        let pair = pty_system.openpty(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })?;

        Ok(Self { pair, reader: None })
    }

    pub fn spawn_command(&mut self, command: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut cmd = CommandBuilder::new(command);
        cmd.env("TERM", "xterm-256color");
        let _child = self.pair.slave.spawn_command(cmd)?;
        Ok(())
    }

    pub fn writer(&mut self) -> Result<Box<dyn Write + Send>, Box<dyn std::error::Error>> {
        Ok(self.pair.master.take_writer()?)
    }

    pub fn reader(&mut self) -> Result<&mut Box<dyn Read + Send>, Box<dyn std::error::Error>> {
        if self.reader.is_none() {
            self.reader = Some(self.pair.master.try_clone_reader()?);
        }
        Ok(self.reader.as_mut().unwrap())
    }

    pub fn write(&mut self, data: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut writer = self.writer()?;
        writer.write_all(data.as_bytes())?;
        writer.flush()?;
        Ok(())
    }

    pub fn read(&mut self, timeout_ms: u64) -> Result<String, Box<dyn std::error::Error>> {
        let reader = self.reader()?;

        if timeout_ms > 0 {
            let start = std::time::Instant::now();
            let mut buffer = Vec::new();
            while start.elapsed().as_millis() < timeout_ms as u128 {
                let mut temp = [0u8; 4096];
                match reader.read(&mut temp) {
                    Ok(0) => break,
                    Ok(n) => buffer.extend_from_slice(&temp[..n]),
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(std::time::Duration::from_millis(10));
                        continue;
                    }
                    Err(e) => return Err(Box::new(e)),
                }
                if !buffer.is_empty() {
                    break;
                }
            }
            return Ok(String::from_utf8_lossy(&buffer).to_string());
        }

        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer)?;
        Ok(String::from_utf8_lossy(&buffer).to_string())
    }

    pub fn resize(&mut self, rows: u16, cols: u16) -> Result<(), Box<dyn std::error::Error>> {
        self.pair.master.resize(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })?;
        Ok(())
    }
}

pub struct PtyReader {
    buffer: Vec<u8>,
    closed: bool,
}

impl PtyReader {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            closed: false,
        }
    }

    pub fn read_from(&mut self, reader: &mut Box<dyn Read + Send>) -> std::io::Result<usize> {
        let mut temp = [0u8; 4096];
        match reader.read(&mut temp) {
            Ok(0) => {
                self.closed = true;
                Ok(0)
            }
            Ok(n) => {
                self.buffer.extend_from_slice(&temp[..n]);
                Ok(n)
            }
            Err(e) => Err(e),
        }
    }

    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    pub fn string(&self) -> String {
        String::from_utf8_lossy(&self.buffer).to_string()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

impl Default for PtyReader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pty_reader_new() {
        let reader = PtyReader::new();
        assert!(reader.buffer().is_empty());
    }

    #[test]
    fn test_pty_reader_string() {
        let reader = PtyReader::new();
        assert_eq!(reader.string(), "");
    }

    #[test]
    fn test_pty_reader_clear() {
        let mut reader = PtyReader::new();
        reader.clear();
        assert!(reader.buffer().is_empty());
    }
}
