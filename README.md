# ChromatUI

ChromaTUI is a kernel-level terminal UI foundation for AI development workloads. It prioritizes deterministic rendering, measurable runtime behavior, and audit-friendly decision systems over ad-hoc widget magic.

## What You Get

- Deterministic runtime loop with explicit events, commands, and effects
- Diff-based renderer with sparse update optimization and ANSI presentation
- Inline and alt-screen output modes with synchronized update bracketing
- Bayesian/conformal/CUSUM-driven runtime controls for strategy and risk decisions
- Widget and extras layers for application composition

## Design Principles

- **Correctness over cleverness**: predictable terminal state is non-negotiable
- **Deterministic output**: explicit frame -> buffer -> diff -> presenter pipeline
- **Inline-first ergonomics**: preserve scrollback while keeping UI chrome stable
- **Layered architecture**: core -> render -> runtime -> widgets, with clear boundaries
- **Evidence before claims**: conformance and validation are tracked in-repo

## Workspace Crates

| Crate | Purpose |
|---|---|
| `chromatui` | Public facade and re-exports |
| `chromatui-core` | Terminal lifecycle, events, capabilities, writer |
| `chromatui-render` | Content/buffer diffing, regions, ANSI presenter |
| `chromatui-runtime` | Deterministic event loop and frame pipeline |
| `chromatui-layout` | Flex and grid layout primitives |
| `chromatui-style` | Colors, themes, style escape generation |
| `chromatui-text` | Text spans, rope structures |
| `chromatui-widgets` | Focus, input, list/table, palette, modal utilities |
| `chromatui-extras` | Optional add-ons (including deterministic visual FX screen) |
| `chromatui-harness` | Snapshot and harness support |
| `chromatui-pty` | PTY utilities for terminal testing |
| `chromatui-simd` | Safe performance helpers |
| `chromatui-algorithms` | Bayesian/diff/control/statistical algorithm library |

## Architecture

```text
Input Layer -> Runtime Loop -> Render Kernel -> Output Layer

TerminalSession (core)
  -> Event stream
  -> update(Event) => (Model', Cmd)
  -> view(Model) => Frame
  -> Frame -> Buffer/Content -> BufferDiff
  -> Presenter -> ANSI bytes
  -> TerminalWriter flush
```

High-level stack:

```text
Application
  Widgets
    Layout
      Runtime
        Render
          Core
```

For full equations and algorithm catalog, see `docs/architecture.md`.
For implementation evidence and verification traces, see `docs/conformance-matrix.md`.

## Notable Runtime/Math Systems

- Bayesian command scoring and evidence ledgers
- Bayesian diff strategy selection with posterior change-rate tracking
- BOCPD resize regime detection and Bayes-factor coalescing evidence
- Conformal risk gating and residual alerting
- E-process + adaptive betting (GRAPA-style updates)
- CUSUM drift detection and hover stabilization
- VOI-driven expensive sampling decisions
- PI/PID pacing control with MPC objective evaluation support
- Count-Min sketch + W-TinyLFU admission primitives and PAC-Bayes bound helper

## Quick Start

Prerequisites:

- Rust stable toolchain
- Cargo

Build workspace:

```bash
cargo check
```

Run tests:

```bash
cargo test
```

Run visual validation example:

```bash
cargo run -p chromatui-runtime --example visual_validation
```

Run render benchmarks:

```bash
cargo bench -p chromatui-render --bench diff_bench -- --sample-size 10
```

## API Entry Point

Use the facade crate in `crates/chromatui/src/lib.rs` for re-exported runtime/core/render primitives:

```rust
use chromatui::{DeterministicRuntime, DiffRenderer, Event, OutputMode, TerminalWriter};
```

Then build your model loop around `Event -> update -> view` and render through the runtime pipeline.

## Documentation Map

- `docs/architecture.md`: full architecture + algorithm reference
- `docs/conformance-matrix.md`: PASS/PARTIAL/MISSING matrix and verification evidence
- `docs/plans/2026-02-22-kernel-tui-foundation-design.md`: foundation design plan
- `docs/plans/2026-02-22-kernel-tui-foundation.md`: implementation plan

## License

MIT License with OpenAI/Anthropic rider.

- Primary license: `LICENSE`
- Rider: `LICENSE-OPENAI-ANTHROPIC-RIDER.md`

Copyright (c) 2026 Mike Holownych
