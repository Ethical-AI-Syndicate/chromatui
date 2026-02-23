pub fn count_nonzero(bytes: &[u8]) -> usize {
    let mut count = 0usize;
    let mut i = 0usize;

    while i + 8 <= bytes.len() {
        let chunk = &bytes[i..i + 8];
        count += usize::from(chunk[0] != 0)
            + usize::from(chunk[1] != 0)
            + usize::from(chunk[2] != 0)
            + usize::from(chunk[3] != 0)
            + usize::from(chunk[4] != 0)
            + usize::from(chunk[5] != 0)
            + usize::from(chunk[6] != 0)
            + usize::from(chunk[7] != 0);
        i += 8;
    }

    while i < bytes.len() {
        count += usize::from(bytes[i] != 0);
        i += 1;
    }

    count
}

pub fn diff_mask_eq(lhs: &[u8], rhs: &[u8]) -> Vec<u8> {
    let len = lhs.len().min(rhs.len());
    let mut mask = Vec::with_capacity(len);

    let mut i = 0usize;
    while i + 8 <= len {
        let l = &lhs[i..i + 8];
        let r = &rhs[i..i + 8];
        mask.push(u8::from(l[0] != r[0]));
        mask.push(u8::from(l[1] != r[1]));
        mask.push(u8::from(l[2] != r[2]));
        mask.push(u8::from(l[3] != r[3]));
        mask.push(u8::from(l[4] != r[4]));
        mask.push(u8::from(l[5] != r[5]));
        mask.push(u8::from(l[6] != r[6]));
        mask.push(u8::from(l[7] != r[7]));
        i += 8;
    }

    while i < len {
        mask.push(u8::from(lhs[i] != rhs[i]));
        i += 1;
    }

    mask
}

pub fn sum_u16(values: &[u16]) -> u64 {
    let mut acc = 0u64;
    let mut i = 0usize;

    while i + 8 <= values.len() {
        let chunk = &values[i..i + 8];
        acc += u64::from(chunk[0])
            + u64::from(chunk[1])
            + u64::from(chunk[2])
            + u64::from(chunk[3])
            + u64::from(chunk[4])
            + u64::from(chunk[5])
            + u64::from(chunk[6])
            + u64::from(chunk[7]);
        i += 8;
    }

    while i < values.len() {
        acc += u64::from(values[i]);
        i += 1;
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_nonzero_matches_scalar() {
        let data = [0, 1, 2, 0, 0, 5, 6, 0, 3, 9];
        assert_eq!(count_nonzero(&data), 6);
    }

    #[test]
    fn diff_mask_identifies_changes() {
        let lhs = [1, 2, 3, 4, 5];
        let rhs = [1, 9, 3, 7, 5];
        assert_eq!(diff_mask_eq(&lhs, &rhs), vec![0, 1, 0, 1, 0]);
    }

    #[test]
    fn sum_u16_matches_expected() {
        let data = [1u16, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(sum_u16(&data), 45);
    }
}
