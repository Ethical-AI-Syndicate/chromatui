use std::io::Write;
use std::mem::size_of;
use std::time::Duration;
use std::time::Instant;

use crate::backend::TerminalSession;
use crate::types::Event;

pub trait Model: Sized + Default {
    fn update(&mut self, event: Event) -> Cmd<Self>;
}

pub type Cmd<M> = Option<Effect<M>>;

#[derive(Default)]
pub enum Effect<M: Model> {
    #[default]
    None,
    Quit,
    Batch(Vec<Effect<M>>),
    Subscribe(Subscription<M>),
    Timeout(Duration, Event),
}

impl<M: Model> Effect<M> {
    pub fn none() -> Self {
        Effect::None
    }

    pub fn quit() -> Self {
        Effect::Quit
    }

    pub fn batch(effects: Vec<Effect<M>>) -> Self {
        if effects.is_empty() {
            Effect::None
        } else {
            Effect::Batch(effects)
        }
    }

    pub fn subscribe(sub: Subscription<M>) -> Self {
        Effect::Subscribe(sub)
    }

    pub fn timeout(delay: Duration, event: Event) -> Self {
        Effect::Timeout(delay, event)
    }
}

pub struct Subscription<M: Model> {
    pub kind: SubscriptionKind,
    _phantom: std::marker::PhantomData<M>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubscriptionKind {
    Tick(Duration),
    Resize,
    Mouse,
    Custom(String),
}

impl<M: Model> Subscription<M> {
    pub fn tick(interval: Duration) -> Self {
        Self {
            kind: SubscriptionKind::Tick(interval),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn resize() -> Self {
        Self {
            kind: SubscriptionKind::Resize,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn mouse() -> Self {
        Self {
            kind: SubscriptionKind::Mouse,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn custom(id: &str) -> Self {
        Self {
            kind: SubscriptionKind::Custom(id.to_string()),
            _phantom: std::marker::PhantomData,
        }
    }
}

pub struct Program<M: Model> {
    model: M,
    subscriptions: Vec<Subscription<M>>,
    timers: Vec<(Instant, Event)>,
    last_tick_fired: Option<Instant>,
    session: TerminalSession,
    running: bool,
}

impl<M: Model> Program<M> {
    pub fn new() -> Self {
        Self {
            model: M::default(),
            subscriptions: Vec::new(),
            timers: Vec::new(),
            last_tick_fired: None,
            session: TerminalSession::new(),
            running: false,
        }
    }

    pub fn with_model(model: M) -> Self {
        Self {
            model,
            subscriptions: Vec::new(),
            timers: Vec::new(),
            last_tick_fired: None,
            session: TerminalSession::new(),
            running: false,
        }
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn subscribe(&mut self, sub: Subscription<M>) {
        self.subscriptions.push(sub);
    }

    pub fn run<F>(&mut self, view: F) -> std::io::Result<()>
    where
        F: Fn(&M) -> Frame,
    {
        self.session.start()?;
        self.running = true;

        let result = (|| {
            while self.running {
                let event = self.poll_event(Duration::from_millis(16))?;

                let cmd = self.model.update(event);
                self.run_cmd(cmd);

                let frame = view(&self.model);
                self.render(frame)?;
            }

            Ok(())
        })();

        let stop_result = self.session.stop();
        result.and(stop_result)
    }

    fn poll_event(&mut self, timeout: Duration) -> std::io::Result<Event> {
        let now = Instant::now();

        if let Some(idx) = self.timers.iter().position(|(at, _)| *at <= now) {
            let (_, event) = self.timers.remove(idx);
            return Ok(event);
        }

        let next_tick_due = self
            .subscriptions
            .iter()
            .filter_map(|s| match s.kind {
                SubscriptionKind::Tick(interval) => Some(interval),
                _ => None,
            })
            .min();

        let tick_wait = if let Some(interval) = next_tick_due {
            let last = self
                .last_tick_fired
                .unwrap_or_else(|| now.checked_sub(interval).unwrap_or(now));
            let due = last.checked_add(interval).unwrap_or(now);
            due.saturating_duration_since(now)
        } else {
            timeout
        };

        let timer_wait = self
            .timers
            .iter()
            .map(|(at, _)| at.saturating_duration_since(now))
            .min()
            .unwrap_or(timeout);

        let wait = timeout.min(tick_wait).min(timer_wait);

        if let Some(event) = self.session.poll_event(wait) {
            return Ok(event);
        }

        let now_after_poll = Instant::now();
        if let Some(idx) = self.timers.iter().position(|(at, _)| *at <= now_after_poll) {
            let (_, event) = self.timers.remove(idx);
            return Ok(event);
        }

        if let Some(interval) = next_tick_due {
            let last = self.last_tick_fired.unwrap_or_else(|| {
                now_after_poll
                    .checked_sub(interval)
                    .unwrap_or(now_after_poll)
            });
            if now_after_poll.saturating_duration_since(last) >= interval {
                self.last_tick_fired = Some(now_after_poll);
                return Ok(Event::Tick);
            }
        }

        Ok(Event::Tick)
    }

    fn run_cmd(&mut self, cmd: Cmd<M>) {
        if let Some(effect) = cmd {
            match effect {
                Effect::None => {}
                Effect::Quit => {
                    self.running = false;
                }
                Effect::Batch(effects) => {
                    for effect in effects {
                        self.run_cmd(Some(effect));
                    }
                }
                Effect::Subscribe(sub) => {
                    if !self.subscriptions.iter().any(|s| s.kind == sub.kind) {
                        self.subscriptions.push(sub);
                    }
                }
                Effect::Timeout(delay, event) => {
                    let at = Instant::now()
                        .checked_add(delay)
                        .unwrap_or_else(Instant::now);
                    self.timers.push((at, event));
                }
            }
        }
    }

    fn render(&self, frame: Frame) -> std::io::Result<()> {
        let mut out = std::io::stdout();
        for row in 0..frame.height {
            out.write_all(format!("\x1b[{};1H", row + 1).as_bytes())?;
            for col in 0..frame.width {
                let idx = row as usize * frame.width as usize + col as usize;
                let ch = frame.cells.get(idx).map(|c| c.ch).unwrap_or(' ');
                let c = if ch == '\0' { ' ' } else { ch };
                let mut buf = [0u8; 4];
                out.write_all(c.encode_utf8(&mut buf).as_bytes())?;
            }
        }
        out.flush()
    }
}

impl<M: Model> Default for Program<M> {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Frame {
    pub width: u16,
    pub height: u16,
    pub cells: Vec<Cell>,
    link_registry: LinkRegistry,
}

impl Frame {
    pub fn new(width: u16, height: u16) -> Self {
        let capacity = (width as usize) * (height as usize);
        Self {
            width,
            height,
            cells: vec![Cell::default(); capacity],
            link_registry: LinkRegistry::new(),
        }
    }

    pub fn cell(&mut self, x: u16, y: u16) -> Option<&mut Cell> {
        if x < self.width && y < self.height {
            let idx = (y as usize) * (self.width as usize) + (x as usize);
            self.cells.get_mut(idx)
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        for cell in &mut self.cells {
            *cell = Cell::default();
        }
    }

    pub fn link_registry(&mut self) -> &mut LinkRegistry {
        &mut self.link_registry
    }

    pub fn link_registry_ref(&self) -> &LinkRegistry {
        &self.link_registry
    }
}

impl Default for Frame {
    fn default() -> Self {
        Self::new(80, 24)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub ch: char,
    pub fg: Color,
    pub bg: Color,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub link_id: u16,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: '\0',
            fg: Color::default(),
            bg: Color::default(),
            bold: false,
            italic: false,
            underline: false,
            link_id: u16::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct PackedCell16 {
    pub ch: u32,
    pub fg: u32,
    pub bg: u32,
    pub attrs: u16,
    pub link_id: u16,
}

impl PackedCell16 {
    pub fn from_cell(cell: Cell) -> Self {
        let attrs =
            (cell.bold as u16) | ((cell.italic as u16) << 1) | ((cell.underline as u16) << 2);
        Self {
            ch: cell.ch as u32,
            fg: u32::from(cell.fg.0) << 16 | u32::from(cell.fg.1) << 8 | u32::from(cell.fg.2),
            bg: u32::from(cell.bg.0) << 16 | u32::from(cell.bg.1) << 8 | u32::from(cell.bg.2),
            attrs,
            link_id: cell.link_id,
        }
    }
}

const _: [(); 16] = [(); size_of::<PackedCell16>()];

#[derive(Debug, Clone, Default)]
pub struct LinkRegistry {
    links: Vec<String>,
}

impl LinkRegistry {
    pub fn new() -> Self {
        Self { links: Vec::new() }
    }

    pub fn register(&mut self, url: &str) -> u16 {
        if let Some((idx, _)) = self.links.iter().enumerate().find(|(_, u)| *u == url) {
            return idx as u16;
        }
        self.links.push(url.to_string());
        (self.links.len() - 1) as u16
    }

    pub fn get(&self, id: u16) -> Option<&str> {
        self.links.get(id as usize).map(String::as_str)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Color(pub u8, pub u8, pub u8);

impl Color {
    pub fn black() -> Self {
        Color(0, 0, 0)
    }
    pub fn white() -> Self {
        Color(255, 255, 255)
    }
    pub fn red() -> Self {
        Color(255, 0, 0)
    }
    pub fn green() -> Self {
        Color(0, 255, 0)
    }
    pub fn blue() -> Self {
        Color(0, 0, 255)
    }
    pub fn yellow() -> Self {
        Color(255, 255, 0)
    }
    pub fn cyan() -> Self {
        Color(0, 255, 255)
    }
    pub fn magenta() -> Self {
        Color(255, 0, 255)
    }
}

impl Cell {
    pub fn with_char(mut self, c: char) -> Self {
        self.ch = c;
        self
    }

    pub fn with_fg(mut self, fg: Color) -> Self {
        self.fg = fg;
        self
    }

    pub fn with_bg(mut self, bg: Color) -> Self {
        self.bg = bg;
        self
    }

    pub fn with_bold(mut self) -> Self {
        self.bold = true;
        self
    }

    pub fn with_italic(mut self) -> Self {
        self.italic = true;
        self
    }

    pub fn with_underline(mut self) -> Self {
        self.underline = true;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Key;

    #[derive(Debug, Default)]
    struct TestModel {
        counter: i32,
    }

    impl Model for TestModel {
        fn update(&mut self, event: Event) -> Cmd<Self> {
            match event {
                Event::Key(Key::Char('+')) => {
                    self.counter += 1;
                    None
                }
                Event::Key(Key::Char('-')) => {
                    self.counter -= 1;
                    None
                }
                Event::Key(Key::Escape) => Some(Effect::quit()),
                _ => None,
            }
        }
    }

    #[test]
    fn test_program_new() {
        let program: Program<TestModel> = Program::new();
        assert_eq!(program.model.counter, 0);
    }

    #[test]
    fn test_model_update() {
        let mut model = TestModel::default();
        model.update(Event::Key(Key::Char('+')));
        assert_eq!(model.counter, 1);
    }

    #[test]
    fn test_cmd_quit() {
        let mut model = TestModel::default();
        let cmd = model.update(Event::Key(Key::Escape));
        assert!(matches!(cmd, Some(Effect::Quit)));
    }

    #[test]
    fn test_frame_cell() {
        let mut frame = Frame::new(10, 10);
        if let Some(cell) = frame.cell(5, 5) {
            cell.ch = 'X';
        }
        assert_eq!(frame.cell(5, 5).unwrap().ch, 'X');
    }

    #[test]
    fn test_cell_builder() {
        let cell = Cell::default()
            .with_char('A')
            .with_fg(Color::red())
            .with_bg(Color::blue())
            .with_bold();

        assert_eq!(cell.ch, 'A');
        assert_eq!(cell.fg, Color(255, 0, 0));
        assert_eq!(cell.bg, Color(0, 0, 255));
        assert!(cell.bold);
    }

    #[test]
    fn test_effect_batch_empty() {
        let effects: Vec<Effect<TestModel>> = vec![];
        let batch = Effect::batch(effects);
        assert!(matches!(batch, Effect::None));
    }

    #[test]
    fn test_frame_clear() {
        let mut frame = Frame::new(3, 3);
        if let Some(cell) = frame.cell(0, 0) {
            cell.ch = 'X';
        }
        frame.clear();
        assert_eq!(frame.cell(0, 0).unwrap().ch, '\0');
    }

    #[test]
    fn test_link_registry_round_trip() {
        let mut frame = Frame::new(4, 2);
        let id = frame.link_registry().register("https://example.com");
        frame.cell(0, 0).expect("cell should exist").link_id = id;

        let resolved = frame
            .link_registry()
            .get(id)
            .expect("link id should resolve");
        assert_eq!(resolved, "https://example.com");
    }

    #[test]
    fn timeout_effect_schedules_timer_event() {
        let mut program: Program<TestModel> = Program::new();
        program.run_cmd(Some(Effect::timeout(
            Duration::from_millis(0),
            Event::Timer(7),
        )));

        let event = program
            .poll_event(Duration::from_millis(0))
            .expect("poll_event should produce timer event");
        assert_eq!(event, Event::Timer(7));
    }
}
