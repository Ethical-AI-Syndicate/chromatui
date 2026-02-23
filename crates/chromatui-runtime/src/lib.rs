use std::collections::HashMap;
use std::io::Write;
use std::time::{Duration, Instant};

use chromatui_algorithms::bocpd::{Regime, RegimeDetector};
use chromatui_algorithms::conformal::{ConformalAlerter, MondrianBucket, MondrianRiskGate};
use chromatui_algorithms::control::{MpcEvaluator, PidController};
use chromatui_algorithms::cusum::CusumDetector;
use chromatui_algorithms::diff::SummedAreaTable;
use chromatui_algorithms::eprocess::EProcess;
use chromatui_algorithms::fairness::InputFairnessGuard;
use chromatui_algorithms::voi::VoiSampler;
use chromatui_core::{
    Cmd, Effect, Event, Frame, Model, OutputMode, Subscription, SubscriptionKind, TerminalSession,
    TerminalWriter,
};
use chromatui_render::{AnsiPresenter, Buffer, BufferDiff, Content, DiffRenderer, Region};

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
    voi_sampler: VoiSampler,
    now_ms: u64,
    last_strategy: DiffStrategy,
    last_decision: DiffDecisionEvidence,
}

impl FramePipeline {
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            diff: DiffRenderer::new(width, height),
            presenter: AnsiPresenter::new(),
            selector: BayesianDiffSelector::new(),
            voi_sampler: VoiSampler::new(),
            now_ms: 0,
            last_strategy: DiffStrategy::DirtyRow,
            last_decision: DiffDecisionEvidence::default(),
        }
    }

    pub fn update_size(&mut self, width: u16, height: u16) {
        self.diff.update_size(width, height);
    }

    pub fn render_frame(&mut self, frame: &Frame) -> Vec<u8> {
        let content = frame_to_content(frame);
        let buffer = Buffer::from_content(&content);
        let buffer_diff: BufferDiff = self.diff.compute_buffer_diff(&buffer);
        let regions = buffer_diff.regions;

        let total_cells = usize::from(frame.width) * usize::from(frame.height);
        let changed_cells = changed_cells_from_regions(&regions, total_cells);
        let dirty_rows = dirty_rows_from_regions(&regions);

        let decision = self
            .selector
            .decide_strategy(frame.width, frame.height, dirty_rows);
        let mut strategy = decision.strategy;
        self.last_strategy = strategy;
        self.last_decision = decision;

        self.selector.observe(changed_cells, total_cells);

        self.now_ms = self.now_ms.saturating_add(16);
        if self.voi_sampler.should_sample(self.now_ms) && matches!(strategy, DiffStrategy::DirtyRow)
        {
            strategy = DiffStrategy::FullDiff;
            self.last_strategy = strategy;
            self.voi_sampler.mark_sampled(self.now_ms);
        }

        let dense = regions_to_dense(&regions, frame.width as usize, frame.height as usize);
        let sat = SummedAreaTable::from_dense(&dense);
        let density = sat.density(0, 0, frame.width as usize, frame.height as usize);
        if density > 0.9 {
            strategy = DiffStrategy::FullRedraw;
            self.last_strategy = strategy;
        }

        let regions_to_render = match strategy {
            DiffStrategy::FullRedraw => vec![Region::new(0, 0, frame.height, frame.width)],
            DiffStrategy::FullDiff | DiffStrategy::DirtyRow => regions,
        };

        self.presenter.encode_regions(&content, &regions_to_render)
    }

    pub fn last_strategy(&self) -> DiffStrategy {
        self.last_strategy
    }

    pub fn last_decision(&self) -> DiffDecisionEvidence {
        self.last_decision.clone()
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
pub struct DiffDecisionEvidence {
    pub strategy: DiffStrategy,
    pub p95_change_rate: f64,
    pub expected_cost_full: f64,
    pub expected_cost_dirty: f64,
    pub expected_cost_redraw: f64,
}

impl Default for DiffDecisionEvidence {
    fn default() -> Self {
        Self {
            strategy: DiffStrategy::DirtyRow,
            p95_change_rate: 0.05,
            expected_cost_full: 0.0,
            expected_cost_dirty: 0.0,
            expected_cost_redraw: 0.0,
        }
    }
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
        self.decide_strategy(width, height, dirty_rows).strategy
    }

    pub fn decide_strategy(
        &self,
        width: u16,
        height: u16,
        dirty_rows: usize,
    ) -> DiffDecisionEvidence {
        let n = (usize::from(width) * usize::from(height)) as f64;
        let p = self.p95_change_rate();
        let d = dirty_rows as f64;
        let w = f64::from(width);
        let dirty_scan = d * w;

        let full_diff =
            self.c_row * f64::from(height) + self.c_scan * dirty_scan + self.c_emit * p * n;
        let dirty_row = self.c_scan * dirty_scan + self.c_emit * p * n;
        let redraw = self.c_emit * n;

        let strategy = if redraw <= full_diff && redraw <= dirty_row {
            DiffStrategy::FullRedraw
        } else if full_diff <= dirty_row {
            DiffStrategy::FullDiff
        } else {
            DiffStrategy::DirtyRow
        };

        DiffDecisionEvidence {
            strategy,
            p95_change_rate: p,
            expected_cost_full: full_diff,
            expected_cost_dirty: dirty_row,
            expected_cost_redraw: redraw,
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
    pub frame_time_ms: f64,
    pub predicted_frame_ms: f64,
    pub risk_upper_bound_ms: f64,
    pub frame_risky: bool,
    pub input_fairness: f64,
    pub forced_yield: bool,
    pub sequential_alert: bool,
    pub conformal_alert: bool,
    pub p95_change_rate: f64,
    pub expected_cost_full: f64,
    pub expected_cost_dirty: f64,
    pub expected_cost_redraw: f64,
}

#[derive(Debug, Clone, Default)]
pub struct RuntimeEvidenceReport {
    pub frames: Vec<FrameEvidence>,
    pub resize_ledger: Vec<ResizeLedgerEntry>,
}

impl RuntimeEvidenceReport {
    pub fn explain_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();

        for frame in &self.frames {
            let resize_hint = self
                .resize_ledger
                .get(frame.step_index.saturating_sub(1))
                .map(|entry| format!(", lbf={:+.3}, regime={:?}", entry.lbf, entry.regime))
                .unwrap_or_default();

            lines.push(format!(
                "step={} event={:?} strategy={:?} bytes={} frame_ms={:.3} pred_ms={:.3} ub_ms={:.3} risky={} fairness={:.3} yield={} seq_alert={} conf_alert={} p95={:.4} full={:.2} dirty={:.2} redraw={:.2}{}",
                frame.step_index,
                frame.event,
                frame.strategy,
                frame.bytes_emitted,
                frame.frame_time_ms,
                frame.predicted_frame_ms,
                frame.risk_upper_bound_ms,
                frame.frame_risky,
                frame.input_fairness,
                frame.forced_yield,
                frame.sequential_alert,
                frame.conformal_alert,
                frame.p95_change_rate,
                frame.expected_cost_full,
                frame.expected_cost_dirty,
                frame.expected_cost_redraw,
                resize_hint
            ));
        }

        lines
    }

    pub fn explain_text(&self) -> String {
        self.explain_lines().join("\n")
    }
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

        let steady_mean = 200.0;
        let burst_mean = 20.0;
        let p_x_given_steady = exp_pdf(inter_arrival, steady_mean);
        let p_x_given_burst = exp_pdf(inter_arrival, burst_mean);
        let regime_prior = match regime {
            Regime::Steady => 2.0,
            Regime::Transitional => 1.0,
            Regime::Burst => 0.5,
        };
        let lbf = ((p_x_given_steady * regime_prior) / p_x_given_burst.max(1e-12)).log10();
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

fn exp_pdf(x: f64, mean: f64) -> f64 {
    let lambda = 1.0 / mean.max(1e-9);
    lambda * (-lambda * x.max(0.0)).exp()
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

fn regions_to_dense(regions: &[Region], width: usize, height: usize) -> Vec<Vec<bool>> {
    let mut dense = vec![vec![false; width]; height];
    for r in regions {
        let y0 = r.start_row as usize;
        let y1 = r.end_row as usize;
        let x0 = r.start_col as usize;
        let x1 = r.end_col as usize;
        for row in dense.iter_mut().take(y1.min(height)).skip(y0.min(height)) {
            for cell in row.iter_mut().take(x1.min(width)).skip(x0.min(width)) {
                *cell = true;
            }
        }
    }
    dense
}

fn dirty_rows_from_regions(regions: &[Region]) -> usize {
    regions
        .iter()
        .map(|r| usize::from(r.end_row.saturating_sub(r.start_row)))
        .sum()
}

pub fn frame_to_content(frame: &Frame) -> Content {
    let mut lines = Vec::with_capacity(frame.height as usize);
    let mut links = Vec::with_capacity(frame.height as usize);

    for y in 0..frame.height {
        let mut line = String::with_capacity(frame.width as usize);
        let mut line_links = Vec::with_capacity(frame.width as usize);
        for x in 0..frame.width {
            let idx = (y as usize) * (frame.width as usize) + (x as usize);
            let cell = frame.cells.get(idx).copied().unwrap_or_default();
            let ch = cell.ch;
            line.push(if ch == '\0' { ' ' } else { ch });
            let link = if cell.link_id == u16::MAX {
                None
            } else {
                frame
                    .link_registry_ref()
                    .get(cell.link_id)
                    .map(|s| s.to_string())
            };
            line_links.push(link);
        }
        lines.push(line);
        links.push(line_links);
    }

    Content::from_lines_with_links(lines, links)
}

pub struct DeterministicRuntime<M: Model, W: Write> {
    model: M,
    pipeline: FramePipeline,
    writer: TerminalWriter<W>,
    subscriptions: Vec<Subscription<M>>,
    subscription_last_fire_ms: HashMap<usize, u64>,
    resize_coalescer: ResizeCoalescer,
    risk_gate: MondrianRiskGate,
    frame_budget_ms: f64,
    predicted_frame_ms: f64,
    pacing_pi: PidController,
    mpc_eval: MpcEvaluator,
    fairness_guard: InputFairnessGuard,
    frame_cusum: CusumDetector,
    frame_eprocess: EProcess,
    frame_conformal: ConformalAlerter,
    force_yield: bool,
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
            subscription_last_fire_ms: HashMap::new(),
            resize_coalescer: ResizeCoalescer::new(),
            risk_gate: MondrianRiskGate::new(0.1),
            frame_budget_ms: 16.7,
            predicted_frame_ms: 16.7,
            pacing_pi: PidController::pi_default(),
            mpc_eval: MpcEvaluator::new(8, 0.1),
            fairness_guard: InputFairnessGuard::new(),
            frame_cusum: CusumDetector::with_params(0.5, 5.0, 0.0),
            frame_eprocess: EProcess::with_params(0.1, 0.0),
            frame_conformal: ConformalAlerter::new(0.1),
            force_yield: false,
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
        let frame_start = Instant::now();
        let ansi = self.pipeline.render_frame(&frame);
        let strategy = self.pipeline.last_strategy();
        let decision = self.pipeline.last_decision();
        self.writer.write_frame(&ansi)?;
        let observed_frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;

        let bucket = MondrianBucket {
            mode: format!("{:?}", self.writer.mode()),
            diff: format!("{:?}", strategy),
            size: frame_size_bucket(frame.width, frame.height).to_string(),
        };
        self.risk_gate
            .update(bucket.clone(), observed_frame_ms, self.predicted_frame_ms);
        let ub_ms = self
            .risk_gate
            .risk_upper_bound(&bucket, self.predicted_frame_ms);
        let risky = self
            .risk_gate
            .is_risky(&bucket, self.predicted_frame_ms, self.frame_budget_ms);

        self.predicted_frame_ms = 0.8 * self.predicted_frame_ms + 0.2 * observed_frame_ms;
        let error = self.frame_budget_ms - observed_frame_ms;
        let control = self.pacing_pi.update(error, 1.0 / 60.0);
        self.frame_budget_ms = (self.frame_budget_ms - 0.05 * control).clamp(8.0, 33.3);

        let _mpc_cost = self.mpc_eval.objective(
            &[self.predicted_frame_ms; 8],
            self.frame_budget_ms,
            &[control; 8],
        );
        let fairness = self.fairness_guard.fairness();

        let delta = observed_frame_ms - self.frame_budget_ms;
        let cusum_alarm = self.frame_cusum.update(delta);
        let wealth = self
            .frame_eprocess
            .update(delta / self.frame_budget_ms.max(1e-6));
        let e_alarm = wealth >= 20.0;
        let sequential_alert = (cusum_alarm && e_alarm) || e_alarm;

        self.frame_conformal
            .update(observed_frame_ms, self.predicted_frame_ms);
        let conformal_alert = self
            .frame_conformal
            .alerts(observed_frame_ms, self.predicted_frame_ms);

        self.step_counter = self.step_counter.saturating_add(1);
        self.evidence.frames.push(FrameEvidence {
            step_index: self.step_counter,
            event: observed_event,
            strategy,
            bytes_emitted: ansi.len(),
            frame_time_ms: observed_frame_ms,
            predicted_frame_ms: self.predicted_frame_ms,
            risk_upper_bound_ms: ub_ms,
            frame_risky: risky,
            input_fairness: fairness,
            forced_yield: self.force_yield,
            sequential_alert,
            conformal_alert,
            p95_change_rate: decision.p95_change_rate,
            expected_cost_full: decision.expected_cost_full,
            expected_cost_dirty: decision.expected_cost_dirty,
            expected_cost_redraw: decision.expected_cost_redraw,
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
        let timeout_ms = timeout.as_millis().max(1) as u64;
        let mut now_ms = 0u64;
        while self.running && steps < max_steps {
            let mut event = self
                .next_subscription_event(now_ms)
                .or_else(|| source.next_event(timeout))
                .unwrap_or(Event::Tick);
            self.fairness_guard.observe(event_source_name(&event));
            self.force_yield = self.fairness_guard.should_yield(timeout.as_millis() as u64);
            if self.force_yield {
                std::thread::yield_now();
            }

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
                Effect::Subscribe(sub) => {
                    if !self.subscriptions.iter().any(|s| s.kind == sub.kind) {
                        self.subscriptions.push(sub);
                    }
                }
                Effect::Timeout(_, _) => {}
            }
        }
    }

    pub fn into_writer(self) -> W {
        self.writer.into_inner()
    }

    fn next_subscription_event(&mut self, now_ms: u64) -> Option<Event> {
        for (idx, sub) in self.subscriptions.iter().enumerate() {
            if let SubscriptionKind::Tick(interval) = &sub.kind {
                let interval_ms = interval.as_millis().max(1) as u64;
                let last = self
                    .subscription_last_fire_ms
                    .get(&idx)
                    .copied()
                    .unwrap_or(0);
                if now_ms.saturating_sub(last) >= interval_ms {
                    self.subscription_last_fire_ms.insert(idx, now_ms);
                    return Some(Event::Tick);
                }
            }
        }
        None
    }
}

fn event_source_name(event: &Event) -> &'static str {
    match event {
        Event::Key(_) => "key",
        Event::Mouse(_) => "mouse",
        Event::Resize(_, _) => "resize",
        Event::Timer(_) => "timer",
        Event::Tick => "tick",
    }
}

fn frame_size_bucket(width: u16, height: u16) -> &'static str {
    let n = usize::from(width) * usize::from(height);
    if n < 2_000 {
        "small"
    } else if n < 8_000 {
        "medium"
    } else {
        "large"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chromatui_core::{Cell, Color, Key};
    use std::sync::{Mutex, OnceLock};

    fn writer_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

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
        let _guard = writer_test_lock()
            .lock()
            .expect("writer test lock should not be poisoned");
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
    fn duplicate_subscriptions_are_deduplicated() {
        let _guard = writer_test_lock()
            .lock()
            .expect("writer test lock should not be poisoned");
        let writer = TerminalWriter::try_new(Vec::<u8>::new(), OutputMode::Inline)
            .expect("writer should be created");
        let mut rt = DeterministicRuntime::new(LoopModel::default(), writer, 6, 1);

        rt.step(Event::Key(Key::Char('s')), view)
            .expect("first subscribe should succeed");
        rt.step(Event::Key(Key::Char('s')), view)
            .expect("duplicate subscribe should succeed");

        assert_eq!(rt.subscriptions_len(), 1);
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
        let _guard = writer_test_lock()
            .lock()
            .expect("writer test lock should not be poisoned");
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
        let decision = sel.decide_strategy(80, 24, 2);
        assert!(matches!(
            strategy,
            DiffStrategy::FullDiff | DiffStrategy::DirtyRow | DiffStrategy::FullRedraw
        ));
        assert!((0.0..=1.0).contains(&decision.p95_change_rate));
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
        let _guard = writer_test_lock()
            .lock()
            .expect("writer test lock should not be poisoned");
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

    #[test]
    fn evidence_explain_text_is_human_readable() {
        let report = RuntimeEvidenceReport {
            frames: vec![FrameEvidence {
                step_index: 1,
                event: Event::Tick,
                strategy: DiffStrategy::DirtyRow,
                bytes_emitted: 42,
                frame_time_ms: 7.5,
                predicted_frame_ms: 8.0,
                risk_upper_bound_ms: 12.0,
                frame_risky: false,
                input_fairness: 1.0,
                forced_yield: false,
                p95_change_rate: 0.12,
                expected_cost_full: 20.0,
                expected_cost_dirty: 10.0,
                expected_cost_redraw: 30.0,
                sequential_alert: false,
                conformal_alert: false,
            }],
            resize_ledger: vec![ResizeLedgerEntry {
                inter_arrival_ms: 20.0,
                lbf: -0.91,
                regime: Regime::Burst,
                applied_now: false,
            }],
        };

        let text = report.explain_text();
        assert!(text.contains("step=1"));
        assert!(text.contains("strategy=DirtyRow"));
        assert!(text.contains("frame_ms=7.500"));
        assert!(text.contains("p95=0.1200"));
        assert!(text.contains("lbf=-0.910"));
        assert!(text.contains("regime=Burst"));
    }

    #[test]
    fn diff_selector_full_cost_uses_dirty_scan_area() {
        let sel = BayesianDiffSelector::new();
        let decision = sel.decide_strategy(100, 20, 2);
        let expected_gap = 20.0;
        let actual_gap = decision.expected_cost_full - decision.expected_cost_dirty;
        assert!((actual_gap - expected_gap).abs() < 1e-9);
    }
}
