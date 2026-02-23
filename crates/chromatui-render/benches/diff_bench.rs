use chromatui_render::{Content, DiffRenderer};
use criterion::{criterion_group, criterion_main, Criterion};

fn build_lines(width: usize, height: usize, fill: u8) -> Vec<String> {
    (0..height)
        .map(|_| std::iter::repeat_n(fill as char, width).collect::<String>())
        .collect()
}

fn bench_diff_identical(c: &mut Criterion) {
    c.bench_function("diff/identical_100x50", |b| {
        b.iter(|| {
            let mut diff = DiffRenderer::new(100, 50);
            let a = Content::from_lines(build_lines(100, 50, b'a'));
            let _ = diff.compute_diff(&a);
            let _ = diff.compute_diff(&a);
        })
    });
}

fn bench_diff_sparse(c: &mut Criterion) {
    c.bench_function("diff/sparse_5pct_100x50", |b| {
        b.iter(|| {
            let mut diff = DiffRenderer::new(100, 50);
            let a = build_lines(100, 50, b'a');
            let mut b2 = a.clone();
            for r in (0..50).step_by(10) {
                b2[r].replace_range(0..1, "Z");
            }
            let a = Content::from_lines(a);
            let b2 = Content::from_lines(b2);
            let _ = diff.compute_diff(&a);
            let _ = diff.compute_diff(&b2);
        })
    });
}

fn bench_diff_dense(c: &mut Criterion) {
    c.bench_function("diff/dense_100x50", |b| {
        b.iter(|| {
            let mut diff = DiffRenderer::new(100, 50);
            let a = Content::from_lines(build_lines(100, 50, b'a'));
            let b2 = Content::from_lines(build_lines(100, 50, b'b'));
            let _ = diff.compute_diff(&a);
            let _ = diff.compute_diff(&b2);
        })
    });
}

criterion_group!(
    benches,
    bench_diff_identical,
    bench_diff_sparse,
    bench_diff_dense
);
criterion_main!(benches);
