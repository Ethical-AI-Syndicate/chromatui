use chromatui::{
    inline::InlineEditor, kernel::StdoutBackend, runtime::state::Runtime, DiffRenderer, Viewport,
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
