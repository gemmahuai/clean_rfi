use clean_rfi::{
    algos::{clean_block, detrend_columns, detrend_rows},
    math::{column_mean, column_var, mask_columns, mask_rows, row_mean, row_var},
};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use faer::prelude::*;
use faer::stats::StandardMat;
use rand::prelude::*;

const BENCH_BLOCK_ROWS: usize = 2048;
const BENCH_BLOCK_COLS: usize = 8192;
const SAMPLES: usize = 10;
const BATCH_SIZE: BatchSize = BatchSize::LargeInput;

macro_rules! bench_mat_fns {
    ($( $fname:ident ),*) => {
        pub fn bench_mat_functions(c: &mut Criterion) {
            let nm = StandardMat {
                nrows: BENCH_BLOCK_ROWS,
                ncols: BENCH_BLOCK_COLS,
            };
            let mut group = c.benchmark_group("Single matrix functions");
            group.sample_size(SAMPLES);
            $(
                group.bench_function(stringify!($fname), |b| {
                    b.iter_batched_ref(
                        || nm.sample(&mut rand::thread_rng()),
                        |data| $fname(data.as_ref()),
                        BATCH_SIZE,
                    )
                });
            )*
            group.finish();
        }
    };
}

bench_mat_fns!(row_mean, column_mean, row_var, column_var);

pub fn bench_mask_rows(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let mask = Col::<f32>::zeros(BENCH_BLOCK_ROWS);
    let mut group = c.benchmark_group("Single matrix functions");
    group.sample_size(SAMPLES);
    group.bench_function("mask_rows", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| mask_rows(data.as_mut(), mask.as_ref()),
            BATCH_SIZE,
        )
    });
    group.finish();
}

pub fn bench_mask_cols(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let mask = Row::<f32>::zeros(BENCH_BLOCK_COLS);
    let mut group = c.benchmark_group("Single matrix functions");
    group.sample_size(SAMPLES);
    group.bench_function("mask_columns", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| mask_columns(data.as_mut(), mask.as_ref()),
            BATCH_SIZE,
        )
    });
    group.finish();
}

pub fn bench_detrend_rows(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let mut group = c.benchmark_group("Single matrix functions");
    group.sample_size(SAMPLES);
    group.bench_function("detrend_rows", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| detrend_rows(data.as_mut(), 4),
            BATCH_SIZE,
        )
    });
    group.finish();
}

pub fn bench_detrend_cols(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let mut group = c.benchmark_group("Single matrix functions");
    group.sample_size(SAMPLES);
    group.bench_function("detrend_columns", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| detrend_columns(data.as_mut(), 4),
            BATCH_SIZE,
        )
    });
    group.finish();
}

pub fn bench_clean_block(c: &mut Criterion) {
    let nm = StandardMat {
        nrows: BENCH_BLOCK_ROWS,
        ncols: BENCH_BLOCK_COLS,
    };
    let mut group = c.benchmark_group("Top-level block cleaning");
    group.sample_size(SAMPLES);
    group.bench_function("clean_block", |b| {
        b.iter_batched_ref(
            || nm.sample(&mut rand::thread_rng()),
            |data| clean_block(data.as_mut()),
            BATCH_SIZE,
        )
    });
}

criterion_group!(
    benches,
    bench_mat_functions,
    bench_mask_rows,
    bench_mask_cols,
    bench_detrend_rows,
    bench_detrend_cols,
    bench_clean_block
);
criterion_main!(benches);
