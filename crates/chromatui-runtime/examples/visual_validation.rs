use chromatui_core::{
    Cell, Cmd, Color, Effect, Event, Frame, Key, Model, OutputMode, TerminalSession, TerminalWriter,
};
use chromatui_runtime::DeterministicRuntime;

#[derive(Default)]
struct VisualModel {
    counter: u32,
    running: bool,
}

impl Model for VisualModel {
    fn update(&mut self, event: Event) -> Cmd<Self> {
        match event {
            Event::Tick => {
                self.counter = self.counter.saturating_add(1);
                self.running = true;
                None
            }
            Event::Key(Key::Char('q')) => Some(Effect::Quit),
            _ => None,
        }
    }
}

fn view(model: &VisualModel) -> Frame {
    let mut frame = Frame::new(16, 2);
    let text = if model.running {
        format!("frame-{:02}", model.counter)
    } else {
        "idle".to_string()
    };
    for (i, ch) in text.chars().enumerate() {
        if let Some(cell) = frame.cell(i as u16, 0) {
            *cell = Cell::default().with_char(ch).with_fg(Color::white());
        }
    }
    frame
}

fn escape_bytes(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &b in bytes {
        match b {
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            0x1b => out.push_str("\\x1b"),
            0x20..=0x7e => out.push(char::from(b)),
            _ => out.push_str(&format!("\\x{:02x}", b)),
        }
    }
    out
}

fn main() -> std::io::Result<()> {
    let mut inline_rt = DeterministicRuntime::new(
        VisualModel::default(),
        TerminalWriter::try_new(Vec::<u8>::new(), OutputMode::Inline)?,
        16,
        2,
    );

    inline_rt.step(Event::Tick, view)?;
    inline_rt.step(Event::Tick, view)?;
    inline_rt.step(Event::Key(Key::Char('q')), view)?;
    let inline_report = inline_rt.evidence_report();
    let inline_bytes = inline_rt.into_writer();

    let mut alt_rt = DeterministicRuntime::new(
        VisualModel::default(),
        TerminalWriter::try_new(Vec::<u8>::new(), OutputMode::AltScreen)?,
        16,
        2,
    );

    alt_rt.step(Event::Tick, view)?;
    alt_rt.step(Event::Resize(16, 2), view)?;
    let alt_report = alt_rt.evidence_report();
    let alt_bytes = alt_rt.into_writer();

    let caps = TerminalSession::new().probe_capabilities();

    println!("VISUAL_VALIDATION_V1");
    println!("inline_ansi={}", escape_bytes(&inline_bytes));
    println!("alt_ansi={}", escape_bytes(&alt_bytes));
    println!(
        "inline_report={}",
        inline_report.explain_text().replace('\n', " | ")
    );
    println!(
        "alt_report={}",
        alt_report.explain_text().replace('\n', " | ")
    );
    println!(
        "capabilities=alt:{:.4},truecolor:{:.4},dark:{:.4},prefers_dark:{}",
        caps.alt_screen_probability,
        caps.truecolor_probability,
        caps.dark_background_probability,
        caps.prefers_dark
    );

    Ok(())
}
