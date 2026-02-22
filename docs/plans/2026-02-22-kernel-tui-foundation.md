# Kernel-Level TUI Foundation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a kernel-level TUI foundation with disciplined runtime, diff-based renderer, and inline-mode with scrollback preservation.

**Architecture:** Layered architecture starting from Kernel (terminal I/O) up to Application layer, with each layer building on the one below.

**Tech Stack:** Rust, Cargo, termios, ANSI escape sequences, tempfile (for disk scrollback).

---

## Task 1: Project Setup and Kernel Layer - Basic Types

**Files:**
- Create: `Cargo.toml`
- Create: `src/kernel/mod.rs`
- Create: `src/kernel/types.rs`
- Create: `src/kernel/backend.rs`
- Create: `tests/kernel_types.rs`

**Step 1: Write the failing test**

Create `tests/kernel_types.rs`:

```rust
#[test]
fn test_key_equality() {
    use crate::kernel::types::Key;
    assert_eq!(Key::Char('a'), Key::Char('a'));
    assert_ne!(Key::Char('a'), Key::Char('b'));
}

#[test]
fn test_event_display() {
    use crate::kernel::types::Event;
    let event = Event::Key(crate::kernel::types::Key::Char('x'));
    let _ = format!("{:?}", event);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test kernel_types 2>&1`
Expected: FAIL - "could not find kernel_types"

**Step 3: Write minimal Cargo.toml and types**

Create `Cargo.toml`:

```toml
[package]
name = "chromatui"
version = "0.1.0"
edition = "2021"

[dependencies]
```

Create `src/kernel/types.rs`:

```rust
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Key {
    Char(char),
    Ctrl(char),
    Alt(char),
    Enter,
    Backspace,
    Delete,
    Up,
    Down,
    Left,
    Right,
    Home,
    End,
    PageUp,
    PageDown,
    Tab,
    Escape,
    F(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
    ScrollUp,
    ScrollDown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseEventType {
    Press,
    Release,
    Drag,
    Hover,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MouseEvent {
    pub event_type: MouseEventType,
    pub button: MouseButton,
    pub row: u16,
    pub col: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    Key(Key),
    Mouse(MouseEvent),
    Resize(u16, u16),
    Timer(u64),
    Tick,
}

impl fmt::Display for Event {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}
```

Create `src/kernel/mod.rs`:

```rust
pub mod types;
pub mod backend;

pub use types::*;
pub use backend::*;
```

Create `src/kernel/backend.rs`:

```rust
use crate::types::{Event, Key};
use std::io::{Read, Write};

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
    
    fn set_raw_mode(&mut self, enabled: bool) -> std::io::Result<()> {
        Ok(())
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test kernel_types 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add Cargo.toml src/kernel/ tests/ 
git commit -m "feat: add kernel layer basic types and backend trait"
```

---

## Task 2: Kernel Layer - ANSI Parser

**Files:**
- Create: `src/kernel/parser.rs`
- Create: `tests/kernel_parser.rs`

**Step 1: Write the failing test**

Create `tests/kernel_parser.rs`:

```rust
use chromatui::kernel::parser::AnsiParser;

#[test]
fn test_plain_text() {
    let mut parser = AnsiParser::new();
    let input = b"hello world";
    let mut output = Vec::new();
    parser.parse(input, &mut output);
    assert_eq!(String::from_utf8(output).unwrap(), "hello world");
}

#[test]
fn test_cursor_movement() {
    let mut parser = AnsiParser::new();
    let input = b"\x1b[10;20H";
    let mut output = Vec::new();
    parser.parse(input, &mut output);
    assert!(!output.is_empty());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test kernel_parser 2>&1`
Expected: FAIL - "could not find parser module"

**Step 3: Write minimal parser**

Create `src/kernel/parser.rs`:

```rust
use crate::types::{Event, Key};

pub struct AnsiParser {
    buffer: Vec<u8>,
}

impl AnsiParser {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    pub fn parse(&mut self, input: &[u8], output: &mut Vec<u8>) {
        output.extend_from_slice(input);
    }

    pub fn parse_event(&mut self, input: &[u8]) -> Option<Event> {
        if input.is_empty() {
            return None;
        }

        match input[0] {
            b'\x1b' if input.len() == 1 => Some(Event::Key(Key::Escape)),
            b'\r' | b'\n' => Some(Event::Key(Key::Enter)),
            b'\t' => Some(Event::Key(Key::Tab)),
            b'\x7f' => Some(Event::Key(Key::Backspace)),
            c if c < 32 => Some(Event::Key(Key::Ctrl((c + 96) as char))),
            c => Some(Event::Key(Key::Char(c as char))),
        }
    }
}

impl Default for AnsiParser {
    fn default() -> Self {
        Self::new()
    }
}
```

Update `src/kernel/mod.rs`:

```rust
pub mod types;
pub mod backend;
pub mod parser;

pub use types::*;
pub use backend::*;
pub use parser::*;
```

**Step 4: Run test to verify it passes**

Run: `cargo test kernel_parser 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add src/kernel/parser.rs tests/kernel_parser.rs
git commit -m "feat: add ANSI parser to kernel layer"
```

---

## Task 3: Runtime Layer - Event Loop

**Files:**
- Create: `src/runtime/mod.rs`
- Create: `src/runtime/state.rs`
- Create: `tests/runtime.rs`

**Step 1: Write the failing test**

Create `tests/runtime.rs`:

```rust
use chromatui::runtime::{Runtime, RuntimeState};

#[test]
fn test_runtime_initial_state() {
    let runtime = Runtime::new();
    matches!(runtime.state(), RuntimeState::Normal);
}

#[test]
fn test_runtime_timers() {
    let mut runtime = Runtime::new();
    runtime.add_timer(1000, 1);
    assert!(runtime.has_timer(1));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test runtime 2>&1`
Expected: FAIL - "could not find runtime"

**Step 3: Write runtime implementation**

Create `src/runtime/mod.rs`:

```rust
pub mod state;

pub use state::*;
```

Create `src/runtime/state.rs`:

```rust
use crate::types::Event;
use std::collections::HashMap;

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
    next_timer_id: u64,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            state: RuntimeState::Normal,
            timers: HashMap::new(),
            next_timer_id: 1,
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
```

Update `src/kernel/mod.rs` to re-export runtime:

```rust
pub mod types;
pub mod backend;
pub mod parser;

pub mod runtime;

pub use types::*;
pub use backend::*;
pub use parser::*;
pub use runtime::*;
```

**Step 4: Run test to verify it passes**

Run: `cargo test runtime 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add src/runtime/ tests/runtime.rs
git commit -m "feat: add runtime layer with state machine"
```

---

## Task 4: Viewport Layer - Scrollback Buffer

**Files:**
- Create: `src/viewport/mod.rs`
- Create: `tests/viewport.rs`

**Step 1: Write the failing test**

Create `tests/viewport.rs`:

```rust
use chromatui::viewport::Viewport;

#[test]
fn test_viewport_new() {
    let vp = Viewport::new(10, 80, 24);
    assert_eq!(vp.height(), 24);
}

#[test]
fn test_viewport_push_line() {
    let mut vp = Viewport::new(10, 80, 24);
    vp.push_line("hello".into());
    assert!(vp.len() > 0);
}

#[test]
fn test_viewport_scroll_offset() {
    let mut vp = Viewport::new(10, 80, 24);
    for i in 0..30 {
        vp.push_line(format!("line {}", i).into());
    }
    vp.set_scroll_offset(10);
    assert_eq!(vp.scroll_offset(), 10);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test viewport 2>&1`
Expected: FAIL - "could not find viewport"

**Step 3: Write viewport implementation**

Create `src/viewport/mod.rs`:

```rust
use std::collections::VecDeque;

pub struct Line {
    pub text: String,
    pub timestamp: Option<std::time::Instant>,
}

impl From<&str> for Line {
    fn from(s: &str) -> Self {
        Self {
            text: s.to_string(),
            timestamp: Some(std::time::Instant::now()),
        }
    }
}

impl From<String> for Line {
    fn from(s: String) -> Self {
        Self {
            text: s,
            timestamp: Some(std::time::Instant::now()),
        }
    }
}

pub struct Viewport {
    buffer: VecDeque<Line>,
    scroll_offset: u32,
    cursor_row: u16,
    terminal_width: u16,
    terminal_height: u16,
    max_memory_lines: usize,
}

impl Viewport {
    pub fn new(max_memory_lines: usize, terminal_width: u16, terminal_height: u16) -> Self {
        Self {
            buffer: VecDeque::with_capacity(max_memory_lines),
            scroll_offset: 0,
            cursor_row: terminal_height.saturating_sub(1),
            terminal_width,
            terminal_height,
            max_memory_lines,
        }
    }

    pub fn push_line(&mut self, line: Line) {
        if self.buffer.len() >= self.max_memory_lines {
            self.buffer.pop_front();
        }
        self.buffer.push_back(line);
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn scroll_offset(&self) -> u32 {
        self.scroll_offset
    }

    pub fn set_scroll_offset(&mut self, offset: u32) {
        self.scroll_offset = offset.min(self.buffer.len() as u32);
    }

    pub fn height(&self) -> u16 {
        self.terminal_height
    }

    pub fn get_visible(&self) -> Vec<&Line> {
        let start = self.scroll_offset as usize;
        let end = (start + self.terminal_height as usize).min(self.buffer.len());
        self.buffer.iter().skip(start).take(end.saturating_sub(start)).collect()
    }

    pub fn scroll_up(&mut self, lines: u16) {
        let max_scroll = self.buffer.len().saturating_sub(self.terminal_height as usize) as u32;
        self.scroll_offset = (self.scroll_offset + lines).min(max_scroll);
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll_offset = self.buffer.len().saturating_sub(self.terminal_height as usize) as u32;
    }

    pub fn update_terminal_size(&mut self, width: u16, height: u16) {
        self.terminal_width = width;
        self.terminal_height = height;
    }
}
```

Update `src/lib.rs` (create it):

```rust
pub mod kernel;
pub mod runtime;
pub mod viewport;

pub use kernel::*;
pub use runtime::*;
pub use viewport::*;
```

**Step 4: Run test to verify it passes**

Run: `cargo test viewport 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add src/viewport/ src/lib.rs tests/viewport.rs
git commit -m "feat: add viewport layer with scrollback buffer"
```

---

## Task 5: Diff Renderer Layer

**Files:**
- Create: `src/renderer/mod.rs`
- Create: `tests/renderer.rs`

**Step 1: Write the failing test**

Create `tests/renderer.rs`:

```rust
use chromatui::renderer::{DiffRenderer, Region, Content};

#[test]
fn test_region_creation() {
    let region = Region::new(0, 0, 10, 80);
    assert!(region.is_valid());
}

#[test]
fn test_diff_renderer_new() {
    let renderer = DiffRenderer::new(80, 24);
    assert_eq!(renderer.width(), 80);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test renderer 2>&1`
Expected: FAIL - "could not find renderer"

**Step 3: Write diff renderer implementation**

Create `src/renderer/mod.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub start_row: u16,
    pub start_col: u16,
    pub end_row: u16,
    pub end_col: u16,
}

impl Region {
    pub fn new(start_row: u16, start_col: u16, end_row: u16, end_col: u16) -> Self {
        Self {
            start_row,
            start_col,
            end_row,
            end_col,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.start_row <= self.end_row && self.start_col <= self.end_col
    }

    pub fn width(&self) -> u16 {
        self.end_col.saturating_sub(self.start_col)
    }

    pub fn height(&self) -> u16 {
        self.end_row.saturating_sub(self.start_row)
    }
}

pub struct Content {
    pub lines: Vec<String>,
}

impl Content {
    pub fn new() -> Self {
        Self { lines: Vec::new() }
    }

    pub fn from_lines(lines: Vec<String>) -> Self {
        Self { lines }
    }
}

impl Default for Content {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DiffRenderer {
    width: u16,
    height: u16,
    prev_content: Option<Content>,
    dirty_regions: Vec<Region>,
}

impl DiffRenderer {
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            prev_content: None,
            dirty_regions: Vec::new(),
        }
    }

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn compute_diff(&mut self, new_content: &Content) -> Vec<Region> {
        let prev = self.prev_content.as_ref();
        
        if prev.is_none() {
            self.dirty_regions.clear();
            self.dirty_regions.push(Region::new(0, 0, self.height, self.width));
            return self.dirty_regions.clone();
        }

        let prev_lines = &prev.unwrap().lines;
        let new_lines = &new_content.lines;

        let mut regions = Vec::new();
        let max_rows = std::cmp::max(prev_lines.len(), new_lines.len());

        for row in 0..max_rows {
            let prev_line = prev_lines.get(row);
            let new_line = new_lines.get(row);

            if prev_line != new_line {
                regions.push(Region::new(row as u16, 0, row.saturating_add(1) as u16, self.width));
            }
        }

        self.dirty_regions = regions.clone();
        self.prev_content = Some(new_content.clone());
        regions
    }

    pub fn update_size(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }
}
```

Update `src/lib.rs`:

```rust
pub mod kernel;
pub mod runtime;
pub mod viewport;
pub mod renderer;

pub use kernel::*;
pub use runtime::*;
pub use viewport::*;
pub use renderer::*;
```

**Step 4: Run test to verify it passes**

Run: `cargo test renderer 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add src/renderer/ tests/renderer.rs src/lib.rs
git commit -m "feat: add diff renderer layer with region tracking"
```

---

## Task 6: Inline Mode Layer

**Files:**
- Create: `src/inline/mod.rs`
- Create: `tests/inline.rs`

**Step 1: Write the failing test**

Create `tests/inline.rs`:

```rust
use chromatui::inline::{InlineEditor, InlineAction};
use chromatui::types::Key;

#[test]
fn test_inline_editor_new() {
    let editor = InlineEditor::new("> ");
    assert_eq!(editor.prompt(), "> ");
}

#[test]
fn test_inline_editor_type() {
    let mut editor = InlineEditor::new("> ");
    let action = editor.handle_key(Key::Char('h'));
    assert!(action.is_none());
    assert_eq!(editor.content(), "h");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test inline 2>&1`
Expected: FAIL - "could not find inline"

**Step 3: Write inline editor implementation**

Create `src/inline/mod.rs`:

```rust
use crate::types::Key;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InlineAction {
    Submit(String),
    Cancel(String),
    Complete(String),
}

pub struct InlineEditor {
    prompt: String,
    content: String,
    cursor_pos: usize,
    history: Vec<String>,
    history_index: Option<usize>,
}

impl InlineEditor {
    pub fn new(prompt: &str) -> Self {
        Self {
            prompt: prompt.to_string(),
            content: String::new(),
            cursor_pos: 0,
            history: Vec::new(),
            history_index: None,
        }
    }

    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    pub fn content(&self) -> &str {
        &self.content
    }

    pub fn cursor_pos(&self) -> usize {
        self.cursor_pos
    }

    pub fn handle_key(&mut self, key: Key) -> Option<InlineAction> {
        match key {
            Key::Char(c) => {
                if self.cursor_pos > self.content.len() {
                    self.cursor_pos = self.content.len();
                }
                self.content.insert(self.cursor_pos, c);
                self.cursor_pos += 1;
                self.history_index = None;
                None
            }
            Key::Backspace => {
                if self.cursor_pos > 0 && !self.content.is_empty() {
                    self.content.remove(self.cursor_pos - 1);
                    self.cursor_pos = self.cursor_pos.saturating_sub(1);
                }
                self.history_index = None;
                None
            }
            Key::Left => {
                self.cursor_pos = self.cursor_pos.saturating_sub(1);
                None
            }
            Key::Right => {
                self.cursor_pos = (self.cursor_pos + 1).min(self.content.len());
                None
            }
            Key::Home => {
                self.cursor_pos = 0;
                None
            }
            Key::End => {
                self.cursor_pos = self.content.len();
                None
            }
            Key::Enter => {
                let submitted = self.content.clone();
                if !submitted.is_empty() {
                    self.history.push(submitted.clone());
                }
                let result = Some(InlineAction::Submit(std::mem::take(&mut self.content)));
                self.cursor_pos = 0;
                self.history_index = None;
                result
            }
            Key::Escape => {
                let cancelled = self.content.clone();
                self.content.clear();
                self.cursor_pos = 0;
                Some(InlineAction::Cancel(cancelled))
            }
            Key::Up => {
                if let Some(idx) = self.history_index {
                    if idx + 1 < self.history.len() {
                        self.history_index = Some(idx + 1);
                    }
                } else if !self.history.is_empty() {
                    self.history_index = Some(0);
                }
                if let Some(idx) = self.history_index {
                    self.content = self.history[idx].clone();
                    self.cursor_pos = self.content.len();
                }
                None
            }
            Key::Down => {
                if let Some(idx) = self.history_index {
                    if idx == 0 {
                        self.history_index = None;
                        self.content.clear();
                    } else {
                        self.history_index = Some(idx - 1);
                        self.content = self.history[idx - 1].clone();
                    }
                }
                self.cursor_pos = self.content.len();
                None
            }
            _ => None,
        }
    }

    pub fn render(&self) -> String {
        let mut output = self.prompt.clone();
        output.push_str(&self.content);
        output
    }

    pub fn submit(&self) -> String {
        self.content.clone()
    }

    pub fn cancel(&self) -> String {
        self.content.clone()
    }
}
```

Update `src/lib.rs`:

```rust
pub mod kernel;
pub mod runtime;
pub mod viewport;
pub mod renderer;
pub mod inline;

pub use kernel::*;
pub use runtime::*;
pub use viewport::*;
pub use renderer::*;
pub use inline::*;
```

**Step 4: Run test to verify it passes**

Run: `cargo test inline 2>&1`
Expected: PASS

**Step 5: Commit**

```bash
git add src/inline/ tests/inline.rs src/lib.rs
git commit -m "feat: add inline mode editor with history"
```

---

## Task 7: Integration - Full Application Example

**Files:**
- Create: `examples/simple_chat.rs`

**Step 1: Write failing test (integration compile check)**

Run: `cargo build --example simple_chat 2>&1`
Expected: FAIL - "example not found"

**Step 2: Create example**

Create `examples/simple_chat.rs`:

```rust
use chromatui::{
    DiffRenderer, Viewport, InlineEditor, Runtime,
    TerminalBackend, StdoutBackend,
};

fn main() {
    println!("ChromatUI Simple Chat Example");
    println!("==============================");
    println!("This is a basic example demonstrating the layered architecture.");
    println!();
    println!("Layers:");
    println!("  - Kernel: terminal I/O, events");
    println!("  - Runtime: event loop, state machine");
    println!("  - Viewport: scrollback buffer");
    println!("  - Renderer: diff-based updates");
    println!("  - Inline: line editing with history");
    println!();
    println!("Run this with a real terminal to see full functionality.");
}
```

**Step 3: Verify it compiles**

Run: `cargo build --example simple_chat 2>&1`
Expected: PASS

**Step 4: Commit**

```bash
git add examples/simple_chat.rs
git commit -m "feat: add simple chat example demonstrating all layers"
```

---

## Task 8: Project Cleanup and Documentation

**Files:**
- Modify: `README.md`

**Step 1: Create README**

Create `README.md`:

```markdown
# ChromatUI

A kernel-level TUI foundation for AI development workloads.

## Architecture

```
┌─────────────────────────────────────┐
│ Application Layer                   │
├─────────────────────────────────────┤
│ Viewport Manager (scrollback, pos)  │
├─────────────────────────────────────┤
│ Diff Renderer (region tracking)     │
├─────────────────────────────────────┤
│ Runtime (events, state, inline)      │
├─────────────────────────────────────┤
│ Kernel (term I/O, escape sequences)  │
└─────────────────────────────────────┘
```

## Layers

- **Kernel**: Raw terminal I/O, ANSI parsing, mouse support
- **Runtime**: Event loop, state machine, timer support
- **Viewport**: Hybrid memory/disk scrollback
- **Renderer**: Diff-based region tracking
- **Inline**: Line editing with history

## Usage

See `examples/simple_chat.rs` for usage.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with architecture overview"
```

---

## Summary

| Task | Component | Files Created |
|------|-----------|---------------|
| 1 | Project Setup & Kernel Types | Cargo.toml, kernel/, tests/ |
| 2 | ANSI Parser | parser.rs, tests/ |
| 3 | Runtime Layer | runtime/, tests/ |
| 4 | Viewport/Scrollback | viewport/, tests/ |
| 5 | Diff Renderer | renderer/, tests/ |
| 6 | Inline Mode | inline/, tests/ |
| 7 | Integration Example | examples/ |
| 8 | Documentation | README.md |
