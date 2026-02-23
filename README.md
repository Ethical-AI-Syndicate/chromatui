# ChromatUI

A kernel-level TUI foundation for AI development workloads with disciplined runtime, diff-based renderer, and inline-mode support that preserves scrollback while keeping UI chrome stable.

## Design Philosophy

- **Correctness over cleverness** — predictable terminal state is non-negotiable
- **Deterministic output** — buffer diffs and explicit presentation over ad-hoc writes
- **Inline first** — preserve scrollback while keeping chrome stable
- **Layered architecture** — core → render → runtime → widgets, no cyclic dependencies
- **Zero-surprise teardown** — RAII cleanup, even when apps crash

## Workspace

| Crate | Purpose |
|-------|---------|
| `chromatui` | Public facade + prelude |
| `chromatui-core` | Terminal lifecycle, events, capabilities |
| `chromatui-render` | Buffer, diff, ANSI presenter |
| `chromatui-style` | Style + theme system |
| `chromatui-text` | Spans, segments, rope editor |
| `chromatui-layout` | Flex + Grid solvers |
| `chromatui-runtime` | Elm/Bubbletea runtime |
| `chromatui-widgets` | Core widget library |
| `chromatui-extras` | Feature-gated add-ons |
| `chromatui-harness` | Reference app + snapshots |
| `chromatui-pty` | PTY test utilities |
| `chromatui-simd` | Optional safe optimizations |

## Architecture

```
┌─────────────────────────────────────┐
│ Application Layer                   │
├─────────────────────────────────────┤
│ Widgets (chromatui-widgets)         │
├─────────────────────────────────────┤
│ Layout (chromatui-layout)           │
├─────────────────────────────────────┤
│ Runtime (chromatui-runtime)         │
├─────────────────────────────────────┤
│ Render (chromatui-render)           │
├─────────────────────────────────────┤
│ Core (chromatui-core)               │
└─────────────────────────────────────┘
```

For the full runtime pipeline and algorithm specification, see `docs/architecture.md`.
Implementation status against that contract is tracked in `docs/conformance-matrix.md`.

## Usage

```rust
use chromatui::{DiffRenderer, Viewport, Runtime, StdoutBackend};

fn main() {
    let mut backend = StdoutBackend::new();
    let mut runtime = Runtime::new();
    let mut viewport = Viewport::new(1000, 80, 24);
    let mut renderer = DiffRenderer::new(80, 24);

    // Your TUI app here
}
```

## License

MIT License, with OpenAI/Anthropic rider.

- Primary license: `LICENSE`
- Rider: `LICENSE-OPENAI-ANTHROPIC-RIDER.md`

Copyright (c) 2026 Mike Holownych
