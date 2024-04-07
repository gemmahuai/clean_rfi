use clean_rfi::{algos, math};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use faer::prelude::*;
use faer::stats::StandardMat;
use rand::prelude::*;

const BENCH_BLOCK_ROWS: usize = 16384;
const BENCH_BLOCK_COLS: usize = 2048;

pub fn row_mean(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let sample: Mat<f32> = nm.sample(&mut rand::thread_rng());
    c.bench_function("row_mean", |b| {
        b.iter(|| math::row_mean(black_box(sample.as_ref())))
    });
}

pub fn column_mean(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let sample: Mat<f32> = nm.sample(&mut rand::thread_rng());
    c.bench_function("column_mean", |b| {
        b.iter(|| math::column_mean(black_box(sample.as_ref())))
    });
}

pub fn column_var(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let sample: Mat<f32> = nm.sample(&mut rand::thread_rng());
    c.bench_function("column_var", |b| {
        b.iter(|| math::column_var(black_box(sample.as_ref())))
    });
}

pub fn row_var(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let sample: Mat<f32> = nm.sample(&mut rand::thread_rng());
    c.bench_function("row_var", |b| {
        b.iter(|| math::row_var(black_box(sample.as_ref())))
    });
}

pub fn detrend_rows(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let mut sample: Mat<f32> = nm.sample(&mut rand::thread_rng());
    c.bench_function("detrend_freq", |b| {
        b.iter(|| algos::detrend_rows_inplace(sample.as_mut(), 4))
    });
}

pub fn detrend_columns(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let mut sample: Mat<f32> = nm.sample(&mut rand::thread_rng());
    c.bench_function("detrend_time", |b| {
        b.iter(|| algos::detrend_columns_inplace(sample.as_mut(), 4))
    });
}

pub fn clean_block(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let mut sample: Mat<f32> = nm.sample(&mut rand::thread_rng());
    c.bench_function("clean_block", |b| {
        b.iter(|| algos::clean_block(sample.as_mut()))
    });
}

criterion_group!(
    benches,
    detrend_rows,
    detrend_columns,
    column_var,
    row_mean,
    column_mean,
    row_var,
    clean_block,
);
criterion_main!(benches);
