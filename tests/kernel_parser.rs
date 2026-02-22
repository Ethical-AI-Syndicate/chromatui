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
