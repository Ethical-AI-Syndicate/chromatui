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
