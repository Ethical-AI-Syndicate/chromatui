use std::collections::HashMap;
use std::io::Write;
use std::time::Duration;

use chromatui_algorithms::bocpd::{Regime, RegimeDetector};
use chromatui_core::{
    Cmd, Effect, Event, Frame, Model, OutputMode, Subscription, TerminalSession, TerminalWriter,
};
use chromatui_render::{AnsiPresenter, Content, DiffRenderer, Region};

pub trait EventSource {
    fn next_event(&mut self, timeout: Duration) -> Option<Event>;
}

impl EventSource for TerminalSession {
    fn next_event(&mut self, timeout: Duration) -> Option<Event> {
        self.poll_event(timeout)
    }
}

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
    selector: BayesianDiffSelector,
    last_strategy: DiffStrategy,
}

impl FramePipeline {
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            diff: DiffRenderer::new(width, height),
            presenter: AnsiPresenter::new(),
            selector: BayesianDiffSelector::new(),
            last_strategy: DiffStrategy::DirtyRow,
        }
    }

    pub fn update_size(&mut self, width: u16, height: u16) {
        self.diff.update_size(width, height);
    }

    pub fn render_frame(&mut self, frame: &Frame) -> Vec<u8> {
        let content = frame_to_content(frame);
        let regions = self.diff.compute_diff(&content);

        let total_cells = usize::from(frame.width) * usize::from(frame.height);
        let changed_cells = changed_cells_from_regions(&regions, total_cells);
        let dirty_rows = dirty_rows_from_regions(&regions);

        let strategy = self
            .selector
            .choose_strategy(frame.width, frame.height, dirty_rows);
        self.last_strategy = strategy;

        self.selector.observe(changed_cells, total_cells);

        let regions_to_render = match strategy {
            DiffStrategy::FullRedraw => vec![Region::new(0, 0, frame.height, frame.width)],
            DiffStrategy::FullDiff | DiffStrategy::DirtyRow => regions,
        };

        self.presenter.encode_regions(&content, &regions_to_render)
    }

    pub fn last_strategy(&self) -> DiffStrategy {
        self.last_strategy
    }
}

impl Default for FramePipeline {
    fn default() -> Self {
        Self::new(80, 24)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffStrategy {
    FullDiff,
    DirtyRow,
    FullRedraw,
}

#[derive(Debug, Clone)]
pub struct BayesianDiffSelector {
    alpha: f64,
    beta: f64,
    decay: f64,
    c_row: f64,
    c_scan: f64,
    c_emit: f64,
}

impl BayesianDiffSelector {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 19.0,
            decay: 0.95,
            c_row: 1.0,
            c_scan: 1.0,
            c_emit: 2.0,
        }
    }

    pub fn observe(&mut self, changed_cells: usize, scanned_cells: usize) {
        self.alpha = self.alpha * self.decay + changed_cells as f64;
        self.beta = self.beta * self.decay + scanned_cells.saturating_sub(changed_cells) as f64;
    }

    pub fn choose_strategy(&self, width: u16, height: u16, dirty_rows: usize) -> DiffStrategy {
        let n = (usize::from(width) * usize::from(height)) as f64;
        let p = self.p95_change_rate();
        let d = dirty_rows as f64;
        let w = f64::from(width);

        let full_diff = self.c_row * f64::from(height) + self.c_scan * n + self.c_emit * p * n;
        let dirty_row = self.c_scan * (d * w) + self.c_emit * p * n;
        let redraw = self.c_emit * n;

        if redraw <= full_diff && redraw <= dirty_row {
            DiffStrategy::FullRedraw
        } else if full_diff <= dirty_row {
            DiffStrategy::FullDiff
        } else {
            DiffStrategy::DirtyRow
        }
    }

    fn p95_change_rate(&self) -> f64 {
        let a = self.alpha.max(1e-9);
        let b = self.beta.max(1e-9);
        let mean = a / (a + b);
        let var = (a * b) / (((a + b) * (a + b)) * (a + b + 1.0));
        (mean + 1.645 * var.sqrt()).clamp(0.0, 1.0)
    }
}

impl Default for BayesianDiffSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ResizeLedgerEntry {
    pub inter_arrival_ms: f64,
    pub lbf: f64,
    pub regime: Regime,
    pub applied_now: bool,
}

#[derive(Debug, Clone)]
pub struct FrameEvidence {
    pub step_index: usize,
    pub event: Event,
    pub strategy: DiffStrategy,
    pub bytes_emitted: usize,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeEvidenceReport {
    pub frames: Vec<FrameEvidence>,
    pub resize_ledger: Vec<ResizeLedgerEntry>,
}

#[derive(Debug, Clone)]
pub struct ResizeCoalescer {
    detector: RegimeDetector,
    last_resize_ms: Option<u64>,
    pending_resize: Option<(u16, u16)>,
    ledger: Vec<ResizeLedgerEntry>,
}

impl ResizeCoalescer {
    pub fn new() -> Self {
        Self {
            detector: RegimeDetector::new(),
            last_resize_ms: None,
            pending_resize: None,
            ledger: Vec::new(),
        }
    }

    pub fn observe_resize_at(&mut self, width: u16, height: u16, now_ms: u64) -> bool {
        let inter_arrival = self
            .last_resize_ms
            .map(|prev| now_ms.saturating_sub(prev) as f64)
            .unwrap_or(200.0);
        self.last_resize_ms = Some(now_ms);

        self.detector.update(inter_arrival);
        let regime = self.detector.regime();

        let regime_lbf = match regime {
            Regime::Steady => 0.8,
            Regime::Transitional => -0.2,
            Regime::Burst => -1.2,
        };
        let cadence_lbf = ((inter_arrival + 1.0) / 50.0).log10();
        let lbf = regime_lbf + cadence_lbf;
        let apply_now = lbf >= 0.0;

        if apply_now {
            self.pending_resize = None;
        } else {
            self.pending_resize = Some((width, height));
        }

        self.ledger.push(ResizeLedgerEntry {
            inter_arrival_ms: inter_arrival,
            lbf,
            regime,
            applied_now: apply_now,
        });

        apply_now
    }

    pub fn take_pending_resize(&mut self) -> Option<Event> {
        self.pending_resize.take().map(|(w, h)| Event::Resize(w, h))
    }

    pub fn last_entry(&self) -> Option<&ResizeLedgerEntry> {
        self.ledger.last()
    }
}

impl Default for ResizeCoalescer {
    fn default() -> Self {
        Self::new()
    }
}

fn changed_cells_from_regions(regions: &[Region], max_cells: usize) -> usize {
    let changed: usize = regions
        .iter()
        .map(|r| {
            usize::from(r.end_row.saturating_sub(r.start_row))
                * usize::from(r.end_col.saturating_sub(r.start_col))
        })
        .sum();
    changed.min(max_cells)
}

fn dirty_rows_from_regions(regions: &[Region]) -> usize {
    regions
        .iter()
        .map(|r| usize::from(r.end_row.saturating_sub(r.start_row)))
        .sum()
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
    resize_coalescer: ResizeCoalescer,
    evidence: RuntimeEvidenceReport,
    step_counter: usize,
    running: bool,
}

impl<M: Model, W: Write> DeterministicRuntime<M, W> {
    pub fn new(model: M, writer: TerminalWriter<W>, width: u16, height: u16) -> Self {
        Self {
            model,
            pipeline: FramePipeline::new(width, height),
            writer,
            subscriptions: Vec::new(),
            resize_coalescer: ResizeCoalescer::new(),
            evidence: RuntimeEvidenceReport::default(),
            step_counter: 0,
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

    pub fn last_diff_strategy(&self) -> DiffStrategy {
        self.pipeline.last_strategy()
    }

    pub fn last_resize_lbf(&self) -> Option<f64> {
        self.resize_coalescer.last_entry().map(|e| e.lbf)
    }

    pub fn evidence_report(&self) -> RuntimeEvidenceReport {
        self.evidence.clone()
    }

    pub fn clear_evidence(&mut self) {
        self.evidence = RuntimeEvidenceReport::default();
        self.step_counter = 0;
    }

    pub fn step<V>(&mut self, event: Event, view: V) -> std::io::Result<()>
    where
        V: Fn(&M) -> Frame,
    {
        if !self.running {
            return Ok(());
        }

        let observed_event = event.clone();

        if let Event::Resize(width, height) = event {
            self.pipeline.update_size(width, height);
        }

        let cmd = self.model.update(event);
        self.run_cmd(cmd);

        let frame = view(&self.model);
        let ansi = self.pipeline.render_frame(&frame);
        let strategy = self.pipeline.last_strategy();
        self.writer.write_frame(&ansi)?;

        self.step_counter = self.step_counter.saturating_add(1);
        self.evidence.frames.push(FrameEvidence {
            step_index: self.step_counter,
            event: observed_event,
            strategy,
            bytes_emitted: ansi.len(),
        });

        Ok(())
    }

    pub fn run_with_source<E, V>(
        &mut self,
        source: &mut E,
        timeout: Duration,
        max_steps: usize,
        view: V,
    ) -> std::io::Result<usize>
    where
        E: EventSource,
        V: Fn(&M) -> Frame,
    {
        let mut steps = 0usize;
        let timeout_ms = timeout.as_millis() as u64;
        let mut now_ms = 0u64;
        while self.running && steps < max_steps {
            let mut event = source.next_event(timeout).unwrap_or(Event::Tick);

            if !matches!(event, Event::Resize(_, _)) {
                if let Some(pending) = self.resize_coalescer.take_pending_resize() {
                    self.step(pending, &view)?;
                    steps += 1;
                    if !self.running || steps >= max_steps {
                        break;
                    }
                }
            }

            if let Event::Resize(w, h) = event {
                if !self.resize_coalescer.observe_resize_at(w, h, now_ms) {
                    if let Some(entry) = self.resize_coalescer.last_entry() {
                        self.evidence.resize_ledger.push(entry.clone());
                    }
                    now_ms = now_ms.saturating_add(timeout_ms);
                    continue;
                }
                if let Some(entry) = self.resize_coalescer.last_entry() {
                    self.evidence.resize_ledger.push(entry.clone());
                }
                event = Event::Resize(w, h);
            }

            self.step(event, &view)?;
            steps += 1;
            now_ms = now_ms.saturating_add(timeout_ms);
        }

        Ok(steps)
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

    struct VecEventSource {
        events: Vec<Event>,
        idx: usize,
    }

    impl VecEventSource {
        fn new(events: Vec<Event>) -> Self {
            Self { events, idx: 0 }
        }
    }

    impl EventSource for VecEventSource {
        fn next_event(&mut self, _timeout: Duration) -> Option<Event> {
            if self.idx >= self.events.len() {
                return None;
            }
            let event = self.events[self.idx].clone();
            self.idx += 1;
            Some(event)
        }
    }

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

    #[test]
    fn run_with_source_drives_event_loop_until_quit() {
        let writer = TerminalWriter::try_new(Vec::<u8>::new(), OutputMode::Inline)
            .expect("writer should be created");
        let mut rt = DeterministicRuntime::new(LoopModel::default(), writer, 6, 1);
        let mut source = VecEventSource::new(vec![
            Event::Tick,
            Event::Resize(6, 1),
            Event::Tick,
            Event::Key(Key::Char('q')),
            Event::Tick,
        ]);

        let steps = rt
            .run_with_source(&mut source, Duration::from_millis(5), 100, view)
            .expect("loop should run successfully");

        assert_eq!(steps, 4);
        assert_eq!(rt.model().count, 2);
        assert!(!rt.is_running());
    }

    #[test]
    fn selector_updates_and_produces_strategy() {
        let mut sel = BayesianDiffSelector::new();
        sel.observe(5, 100);
        let strategy = sel.choose_strategy(80, 24, 2);
        assert!(matches!(
            strategy,
            DiffStrategy::FullDiff | DiffStrategy::DirtyRow | DiffStrategy::FullRedraw
        ));
    }

    #[test]
    fn resize_coalescer_produces_negative_lbf_on_burst() {
        let mut coalescer = ResizeCoalescer::new();
        for i in 0..8 {
            let _ = coalescer.observe_resize_at(100, 40, i * 20);
        }

        let last = coalescer
            .last_entry()
            .expect("resize coalescer should record entries");
        assert!(last.lbf < 0.0);
    }

    #[test]
    fn evidence_report_contains_frame_and_resize_entries() {
        let writer = TerminalWriter::try_new(Vec::<u8>::new(), OutputMode::Inline)
            .expect("writer should be created");
        let mut rt = DeterministicRuntime::new(LoopModel::default(), writer, 6, 1);
        let mut source = VecEventSource::new(vec![
            Event::Resize(6, 1),
            Event::Resize(6, 1),
            Event::Tick,
            Event::Key(Key::Char('q')),
        ]);

        let _ = rt
            .run_with_source(&mut source, Duration::from_millis(20), 20, view)
            .expect("runtime loop should succeed");

        let report = rt.evidence_report();
        assert!(!report.frames.is_empty());
        assert!(!report.resize_ledger.is_empty());
        assert!(report.frames.iter().all(|f| f.step_index > 0));
    }
}
