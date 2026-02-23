use std::time::Duration;

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
    running: bool,
}

impl<M: Model> Program<M> {
    pub fn new() -> Self {
        Self {
            model: M::default(),
            subscriptions: Vec::new(),
            running: false,
        }
    }

    pub fn with_model(model: M) -> Self {
        Self {
            model,
            subscriptions: Vec::new(),
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
        self.running = true;

        while self.running {
            let event = self.poll_event(Duration::from_millis(16))?;

            let cmd = self.model.update(event);
            self.run_cmd(cmd);

            let frame = view(&self.model);
            self.render(frame);
        }

        Ok(())
    }

    fn poll_event(&mut self, _timeout: Duration) -> std::io::Result<Event> {
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
                    self.subscriptions.push(sub);
                }
                Effect::Timeout(_, _) => {}
            }
        }
    }

    fn render(&self, _frame: Frame) {}
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
}

impl Frame {
    pub fn new(width: u16, height: u16) -> Self {
        let capacity = (width as usize) * (height as usize);
        Self {
            width,
            height,
            cells: vec![Cell::default(); capacity],
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
}

impl Default for Frame {
    fn default() -> Self {
        Self::new(80, 24)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Cell {
    pub ch: char,
    pub fg: Color,
    pub bg: Color,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
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
}
