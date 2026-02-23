use std::collections::HashMap;
use std::io::Write;

use chromatui_core::{Cmd, Effect, Event, Frame, Model, OutputMode, Subscription, TerminalWriter};
use chromatui_render::{AnsiPresenter, Content, DiffRenderer};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeState {
    Normal,
    Inline(InlineContext),
    Suspended,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlineContext {
    pub prompt: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timer {
    pub id: u64,
    pub interval_ms: u64,
}

pub struct Runtime {
    state: RuntimeState,
    timers: HashMap<u64, Timer>,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            state: RuntimeState::Normal,
            timers: HashMap::new(),
        }
    }

    pub fn state(&self) -> &RuntimeState {
        &self.state
    }

    pub fn add_timer(&mut self, interval_ms: u64, id: u64) {
        self.timers.insert(id, Timer { id, interval_ms });
    }

    pub fn has_timer(&self, id: u64) -> bool {
        self.timers.contains_key(&id)
    }

    pub fn remove_timer(&mut self, id: u64) {
        self.timers.remove(&id);
    }

    pub fn push_inline_mode(&mut self, prompt: &str) {
        self.state = RuntimeState::Inline(InlineContext {
            prompt: prompt.to_string(),
        });
    }

    pub fn pop_inline_mode(&mut self) -> String {
        let old = std::mem::replace(&mut self.state, RuntimeState::Normal);
        if let RuntimeState::Inline(ctx) = old {
            ctx.prompt
        } else {
            String::new()
        }
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

pub struct FramePipeline {
    diff: DiffRenderer,
    presenter: AnsiPresenter,
}

impl FramePipeline {
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            diff: DiffRenderer::new(width, height),
            presenter: AnsiPresenter::new(),
        }
    }

    pub fn update_size(&mut self, width: u16, height: u16) {
        self.diff.update_size(width, height);
    }

    pub fn render_frame(&mut self, frame: &Frame) -> Vec<u8> {
        let content = frame_to_content(frame);
        let regions = self.diff.compute_diff(&content);
        self.presenter.encode_regions(&content, &regions)
    }
}

impl Default for FramePipeline {
    fn default() -> Self {
        Self::new(80, 24)
    }
}

pub fn frame_to_content(frame: &Frame) -> Content {
    let mut lines = Vec::with_capacity(frame.height as usize);

    for y in 0..frame.height {
        let mut line = String::with_capacity(frame.width as usize);
        for x in 0..frame.width {
            let idx = (y as usize) * (frame.width as usize) + (x as usize);
            let ch = frame.cells.get(idx).map(|c| c.ch).unwrap_or(' ');
            line.push(if ch == '\0' { ' ' } else { ch });
        }
        lines.push(line);
    }

    Content::from_lines(lines)
}

pub struct DeterministicRuntime<M: Model, W: Write> {
    model: M,
    pipeline: FramePipeline,
    writer: TerminalWriter<W>,
    subscriptions: Vec<Subscription<M>>,
    running: bool,
}

impl<M: Model, W: Write> DeterministicRuntime<M, W> {
    pub fn new(model: M, writer: TerminalWriter<W>, width: u16, height: u16) -> Self {
        Self {
            model,
            pipeline: FramePipeline::new(width, height),
            writer,
            subscriptions: Vec::new(),
            running: true,
        }
    }

    pub fn mode(&self) -> OutputMode {
        self.writer.mode()
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn is_running(&self) -> bool {
        self.running
    }

    pub fn stop(&mut self) {
        self.running = false;
    }

    pub fn subscriptions_len(&self) -> usize {
        self.subscriptions.len()
    }

    pub fn step<V>(&mut self, event: Event, view: V) -> std::io::Result<()>
    where
        V: Fn(&M) -> Frame,
    {
        if !self.running {
            return Ok(());
        }

        if let Event::Resize(width, height) = event {
            self.pipeline.update_size(width, height);
        }

        let cmd = self.model.update(event);
        self.run_cmd(cmd);

        let frame = view(&self.model);
        let ansi = self.pipeline.render_frame(&frame);
        self.writer.write_frame(&ansi)
    }

    fn run_cmd(&mut self, cmd: Cmd<M>) {
        if let Some(effect) = cmd {
            match effect {
                Effect::None => {}
                Effect::Quit => self.running = false,
                Effect::Batch(effects) => {
                    for effect in effects {
                        self.run_cmd(Some(effect));
                    }
                }
                Effect::Subscribe(sub) => self.subscriptions.push(sub),
                Effect::Timeout(_, _) => {}
            }
        }
    }

    pub fn into_writer(self) -> W {
        self.writer.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chromatui_core::{Cell, Color, Key};

    #[derive(Default)]
    struct LoopModel {
        count: i32,
    }

    impl Model for LoopModel {
        fn update(&mut self, event: Event) -> Cmd<Self> {
            match event {
                Event::Tick => {
                    self.count += 1;
                    None
                }
                Event::Key(Key::Char('q')) => Some(Effect::Quit),
                Event::Key(Key::Char('s')) => Some(Effect::subscribe(Subscription::tick(
                    std::time::Duration::from_millis(16),
                ))),
                _ => None,
            }
        }
    }

    fn view(model: &LoopModel) -> Frame {
        let mut frame = Frame::new(6, 1);
        let txt = format!("{:>6}", model.count);
        for (i, ch) in txt.chars().enumerate() {
            if let Some(cell) = frame.cell(i as u16, 0) {
                *cell = Cell::default().with_char(ch).with_fg(Color::white());
            }
        }
        frame
    }

    #[test]
    fn frame_pipeline_deterministic_bytes() {
        let mut pipeline = FramePipeline::new(4, 1);
        let mut frame = Frame::new(4, 1);
        for (idx, ch) in ['t', 'e', 's', 't'].iter().enumerate() {
            frame.cell(idx as u16, 0).expect("cell must exist").ch = *ch;
        }

        let first = pipeline.render_frame(&frame);
        let second = pipeline.render_frame(&frame);

        assert_eq!(first, b"\x1b[1;1Htest");
        assert_eq!(second, b"");
    }

    #[test]
    fn loop_enforces_effects_and_writer_output() {
        let writer = TerminalWriter::try_new(Vec::<u8>::new(), OutputMode::Inline)
            .expect("writer should be created");
        let mut rt = DeterministicRuntime::new(LoopModel::default(), writer, 6, 1);

        rt.step(Event::Tick, view)
            .expect("tick step should succeed");
        rt.step(Event::Key(Key::Char('s')), view)
            .expect("subscribe step should succeed");
        rt.step(Event::Key(Key::Char('q')), view)
            .expect("quit step should succeed");

        assert_eq!(rt.model().count, 1);
        assert_eq!(rt.subscriptions_len(), 1);
        assert!(!rt.is_running());

        let bytes = rt.into_writer();
        let rendered = String::from_utf8(bytes).expect("output should be valid UTF-8");
        assert!(rendered.contains("\r"));
        assert!(rendered.contains("1"));
    }

    #[test]
    fn frame_to_content_maps_nul_to_space() {
        let mut frame = Frame::new(3, 1);
        frame.cell(0, 0).expect("cell 0 exists").ch = 'A';
        frame.cell(1, 0).expect("cell 1 exists").ch = '\0';
        frame.cell(2, 0).expect("cell 2 exists").ch = 'Z';

        let content = frame_to_content(&frame);
        assert_eq!(content.lines, vec!["A Z".to_string()]);
    }
}
