use clean_rfi::{algos, math};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use faer::prelude::*;
use faer::stats::StandardMat;
use rand::prelude::*;

const BENCH_BLOCK_ROWS: usize = 2048;
const BENCH_BLOCK_COLS: usize = 15324;

pub fn row_mean(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    c.bench_function("row_mean", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| math::row_mean(data.as_ref()),
            BatchSize::LargeInput,
        )
    });
}

pub fn column_mean(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    c.bench_function("column_mean", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| math::column_mean(data.as_ref()),
            BatchSize::LargeInput,
        )
    });
}

pub fn column_var(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    c.bench_function("column_var", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| math::column_var(data.as_ref()),
            BatchSize::LargeInput,
        )
    });
}

pub fn row_var(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    c.bench_function("row_var", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| math::row_var(data.as_ref()),
            BatchSize::LargeInput,
        )
    });
}

pub fn detrend_rows(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    c.bench_function("detrend_rows", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| algos::detrend_rows(data.as_mut(), 4),
            BatchSize::LargeInput,
        )
    });
}

pub fn detrend_columns(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    c.bench_function("detrend_columns", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| algos::detrend_columns(data.as_mut(), 4),
            BatchSize::LargeInput,
        )
    });
}

pub fn clean_block(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    c.bench_function("clean_block", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| algos::clean_block(data.as_mut()),
            BatchSize::LargeInput,
        )
    });
}

pub fn mask_columns(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let mask = Row::<f32>::zeros(BENCH_BLOCK_COLS);
    c.bench_function("mask_columns", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| math::mask_columns(data.as_mut(), mask.as_ref()),
            BatchSize::LargeInput,
        )
    });
}

pub fn mask_rows(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let mask = Col::<f32>::zeros(BENCH_BLOCK_ROWS);
    c.bench_function("mask_rows", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| math::mask_rows(data.as_mut(), mask.as_ref()),
            BatchSize::LargeInput,
        )
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
    mask_columns,
    mask_rows
);
criterion_main!(benches);
