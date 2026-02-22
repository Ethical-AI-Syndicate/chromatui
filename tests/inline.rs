use chromatui::inline::{InlineAction, InlineEditor};
use chromatui::kernel::Key;

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
