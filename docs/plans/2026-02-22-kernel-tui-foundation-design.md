# Kernel-Level TUI Foundation Design

> **For Claude:** Use writing-plans skill to create implementation plan after this design is committed.

**Goal:** Build a reusable, layered TUI foundation for AI development workloads featuring a disciplined runtime, diff-based renderer, and inline-mode with persistent scrollback.

**Architecture:** Layered architecture with clear separation between Kernel (terminal I/O), Runtime (event loop + state), Diff Renderer (region-based updates), Viewport (scrollback management), and Inline Mode (line editing).

**Tech Stack:** Rust, termios/TTY, ANSI escape sequences, hybrid memory/disk scrollback.

---

## 1. Kernel Layer

### Responsibilities
- Raw terminal I/O via termios/TTY
- Parse ANSI escape sequences (CSI, OSC, DCS)
- Truecolor (24-bit) support
- Mouse protocol handling (SGR, UTF8 modes)
- Input buffering and key parsing

### API
```rust
trait TerminalBackend {
    fn read_event(&self) -> Event;
    fn write(&self, data: &[u8]);
    fn get_size(&self) -> (u16, u16);
    fn set_raw_mode(&self, enabled: bool);
}
```

### Events
```rust
enum Event {
    Key(Key),
    Mouse(MouseEvent),
    Resize(u16, u16),
    Timer(TimerId),
}
```

---

## 2. Runtime Layer

### Responsibilities
- Event loop with poll/select
- State machine for modes (normal, insert, inline)
- Timer/interval support for AI workloads (streaming, polling)
- Input dispatch to handlers
- Flow control for rendering

### API
```rust
struct Runtime {
    event_tx: channel<Event>,
    state: RuntimeState,
    timers: HashMap<TimerId, Timer>,
}

enum RuntimeState {
    Normal,
    Inline(InlineContext),
    Suspended,
}

impl Runtime {
    fn run(&mut self, backend: &mut TerminalBackend);
    fn push_inline_mode(&mut self, prompt: &str);
    fn pop_inline_mode(&mut self) -> String;
}
```

---

## 3. Diff Renderer

### Responsibilities
- Track content regions across frames
- Compute minimal region diffs
- Batch updates for efficient terminal writes
- Handle virtual scrolling for large content

### API
```rust
struct DiffRenderer {
    regions: Vec<Region>,
    prev_content: Content,
    batch_buffer: Vec<u8>,
}

struct Region {
    start_row: u16,
    start_col: u16,
    end_row: u16,
    end_col: u16,
    dirty: bool,
}

impl DiffRenderer {
    fn compute_diff(&mut self, new_content: &Content) -> Vec<Region>;
    fn render_regions(&mut self, regions: &[Region], content: &Content) -> Vec<u8>;
    fn flush(&mut self, backend: &mut impl TerminalBackend);
}
```

---

## 4. Viewport / Scrollback

### Responsibilities
- Hybrid memory + disk scrollback
- Ring buffer for recent lines (hot path)
- File spillover for long sessions
- Virtual cursor positioning
- Scroll offset management

### API
```rust
struct Viewport {
    memory_buffer: RingBuffer<Line>,
    disk_buffer: Option<DiskBackedBuffer>,
    scroll_offset: u32,
    cursor_row: u16,
    terminal_height: u16,
}

impl Viewport {
    fn push_line(&mut self, line: Line);
    fn get_visible(&self) -> &[Line];
    fn scroll_up(&mut self, lines: u16);
    fn scroll_to_bottom(&mut self);
    fn save_to_disk(&mut self) -> io::Result<()>;
}
```

---

## 5. Inline Mode

### Responsibilities
- Line editing with cursor movement
- History (persistent, searchable)
- Completion suggestions
- Prompt management
- Preserve scrollback on exit

### API
```rust
struct InlineEditor {
    content: String,
    cursor_pos: usize,
    history: History,
    prompt: String,
    suggestions: Vec<String>,
}

impl InlineEditor {
    fn handle_input(&mut self, event: Event) -> Option<InlineAction>;
    fn render(&self) -> String;
    fn submit(&self) -> String;
    fn cancel(&self) -> String;
}
```

---

## Layer Dependencies

```
┌─────────────────────────────────────┐
│ Application Layer (AI Chat, etc.)   │
├─────────────────────────────────────┤
│ Viewport Manager (scrollback, pos)  │
├─────────────────────────────────────┤
│ Diff Renderer (region tracking)     │
├─────────────────────────────────────┤
│ Runtime (events, state, inline)    │
├─────────────────────────────────────┤
│ Kernel (term I/O, escape sequences) │
└─────────────────────────────────────┘
```

---

## Testing Strategy

1. **Kernel:** Mock terminal backend for deterministic input/output
2. **Runtime:** State machine property tests
3. **Diff Renderer:** Golden tests with known diff outputs
4. **Viewport:** Memory vs disk buffer consistency tests
5. **Inline:** Integration tests with simulated user input
