use crate::{Content, Region};

pub struct AnsiPresenter {
    cursor: Option<(u16, u16)>,
}

impl AnsiPresenter {
    pub fn new() -> Self {
        Self { cursor: None }
    }

    pub fn encode_regions(&mut self, content: &Content, regions: &[Region]) -> Vec<u8> {
        let mut out = Vec::new();

        for region in regions {
            if !region.is_valid() {
                continue;
            }

            for row in region.start_row..region.end_row {
                let line = content
                    .lines
                    .get(row as usize)
                    .map(String::as_str)
                    .unwrap_or("");
                let line_bytes = line.as_bytes();

                let start = region.start_col as usize;
                let end = region.end_col as usize;
                if start >= line_bytes.len() {
                    self.move_cursor(&mut out, row + 1, region.start_col + 1);
                    continue;
                }

                let end = end.min(line_bytes.len());
                let slice = &line_bytes[start..end];

                self.move_cursor(&mut out, row + 1, region.start_col + 1);
                out.extend_from_slice(slice);
                self.cursor = Some((row + 1, region.start_col + 1 + slice.len() as u16));
            }
        }

        out
    }

    fn move_cursor(&mut self, out: &mut Vec<u8>, row: u16, col: u16) {
        if self.cursor == Some((row, col)) {
            return;
        }
        out.extend_from_slice(format!("\x1b[{};{}H", row, col).as_bytes());
        self.cursor = Some((row, col));
    }
}

impl Default for AnsiPresenter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_cursor_moves_and_line_slices() {
        let mut presenter = AnsiPresenter::new();
        let content = Content::from_lines(vec!["hello".to_string(), "world".to_string()]);
        let regions = vec![Region::new(0, 1, 2, 4)];
        let bytes = presenter.encode_regions(&content, &regions);
        let rendered = String::from_utf8(bytes).expect("presenter output must be UTF-8/ASCII");

        assert_eq!(rendered, "\x1b[1;2Hell\x1b[2;2Horl");
    }

    #[test]
    fn deterministic_for_same_input() {
        let mut presenter = AnsiPresenter::new();
        let content = Content::from_lines(vec!["abc".to_string()]);
        let regions = vec![Region::new(0, 0, 1, 3)];

        let first = presenter.encode_regions(&content, &regions);

        let mut presenter2 = AnsiPresenter::new();
        let second = presenter2.encode_regions(&content, &regions);
        assert_eq!(first, second);
    }
}
