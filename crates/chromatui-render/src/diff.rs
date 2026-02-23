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

#[derive(Debug, Clone)]
pub struct Content {
    pub lines: Vec<String>,
    pub links: Vec<Vec<Option<String>>>,
}

#[derive(Debug, Clone)]
pub struct Buffer {
    pub lines: Vec<String>,
}

impl Buffer {
    pub fn from_content(content: &Content) -> Self {
        Self {
            lines: content.lines.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BufferDiff {
    pub regions: Vec<Region>,
}

impl Content {
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            links: Vec::new(),
        }
    }

    pub fn from_lines(lines: Vec<String>) -> Self {
        let links = lines
            .iter()
            .map(|line| vec![None; line.len()])
            .collect::<Vec<_>>();
        Self { lines, links }
    }

    pub fn from_lines_with_links(lines: Vec<String>, links: Vec<Vec<Option<String>>>) -> Self {
        Self { lines, links }
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
            self.dirty_regions
                .push(Region::new(0, 0, self.height, self.width));
            self.prev_content = Some(new_content.clone());
            return self.dirty_regions.clone();
        }

        let prev_lines = &prev.unwrap().lines;
        let prev_links_all = &prev.unwrap().links;
        let new_lines = &new_content.lines;
        let new_links_all = &new_content.links;

        let mut regions = Vec::new();
        let max_rows = std::cmp::max(prev_lines.len(), new_lines.len());

        for row in 0..max_rows {
            let prev_line = prev_lines.get(row);
            let new_line = new_lines.get(row);

            let prev_links = prev_links_all.get(row);
            let new_links = new_links_all.get(row);
            if prev_line != new_line || prev_links != new_links {
                regions.push(Region::new(
                    row as u16,
                    0,
                    row.saturating_add(1) as u16,
                    self.width,
                ));
            }
        }

        self.dirty_regions = regions.clone();
        self.prev_content = Some(new_content.clone());
        regions
    }

    pub fn compute_buffer_diff(&mut self, buffer: &Buffer) -> BufferDiff {
        let content = Content::from_lines(buffer.lines.clone());
        BufferDiff {
            regions: self.compute_diff(&content),
        }
    }

    pub fn update_size(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }
}
