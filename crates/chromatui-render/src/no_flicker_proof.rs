use crate::diff::{Buffer, Content, DiffRenderer};

pub fn theorem_diff_completeness(old_lines: Vec<String>, new_lines: Vec<String>) -> bool {
    let height = old_lines.len().max(new_lines.len()) as u16;
    let width = old_lines
        .iter()
        .chain(new_lines.iter())
        .map(|l| l.len())
        .max()
        .unwrap_or(0) as u16;

    let mut renderer = DiffRenderer::new(width, height);
    let old = Content::from_lines(old_lines);
    let new = Content::from_lines(new_lines);
    let _ = renderer.compute_buffer_diff(&Buffer::from_content(&old));
    let diff = renderer.compute_buffer_diff(&Buffer::from_content(&new));

    !diff.regions.is_empty() || old.lines == new.lines
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *seed
    }

    fn random_line(seed: &mut u64, len: usize) -> String {
        let mut s = String::with_capacity(len);
        for _ in 0..len {
            let v = (lcg(seed) % 26) as u8;
            s.push((b'a' + v) as char);
        }
        s
    }

    #[test]
    fn counterexample_diff_completeness_holds_for_change() {
        let ok = theorem_diff_completeness(vec!["abc".into()], vec!["axc".into()]);
        assert!(ok);
    }

    #[test]
    fn counterexample_diff_completeness_holds_for_equal() {
        let ok = theorem_diff_completeness(vec!["abc".into()], vec!["abc".into()]);
        assert!(ok);
    }

    #[test]
    fn property_diff_completeness_randomized() {
        let mut seed = 42u64;
        for _ in 0..200 {
            let rows = (lcg(&mut seed) % 8 + 1) as usize;
            let cols = (lcg(&mut seed) % 32 + 1) as usize;
            let mut old_lines = Vec::with_capacity(rows);
            let mut new_lines = Vec::with_capacity(rows);

            for _ in 0..rows {
                let line = random_line(&mut seed, cols);
                old_lines.push(line.clone());

                let mut edited = line;
                if (lcg(&mut seed) & 1) == 1 {
                    let idx = (lcg(&mut seed) as usize) % cols;
                    let mut bytes = edited.into_bytes();
                    bytes[idx] = b'Z';
                    edited = String::from_utf8(bytes).expect("must stay utf8");
                }
                new_lines.push(edited);
            }

            assert!(theorem_diff_completeness(old_lines, new_lines));
        }
    }
}
