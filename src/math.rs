use faer::prelude::*;
use std::simd::prelude::*; // 8 f32s, AVX (not AVX512)

// Conventions:
// indexing is always row,column
// which will be represented in i,j
// the length of these is m,n such that 0 <= i < m and 0 <= j < n
// Faer mats are column-major, so in-order indexing is idx = i + jm

// Arrays are masked with NaNs

#[inline(always)]
fn column_mean_with_counts(mat: MatRef<'_, f32>, mut counts: ColMut<f32>) -> Col<f32> {
    let m = mat.nrows();
    let n = mat.ncols();

    let mut mean = Col::<f32>::zeros(m);

    for j in 0..n {
        zipped!(&mut mean, &mut counts, mat.as_ref().col(j)).for_each(
            |unzipped!(mut mean, mut counts, data)| {
                if !(*data).is_nan() {
                    *mean += *data;
                    *counts += 1.0;
                }
            },
        );
    }

    zipped!(&mut mean, counts.as_ref()).for_each(|unzipped!(mut mean, counts)| {
        *mean /= *counts;
    });

    mean
}

/// Compute the column mean, ignoring masked (NaN) values
pub fn column_mean(mat: MatRef<'_, f32>) -> Col<f32> {
    column_mean_with_counts(mat, Col::zeros(mat.nrows()).as_mut())
}

/// Compute the column variance, ignoring masked (NaN) values, with a precomputed mean
#[inline(always)]
fn column_var_with_mean_counts(
    mat: MatRef<'_, f32>,
    mean: ColRef<'_, f32>,
    mut counts: ColMut<'_, f32>,
) -> Col<f32> {
    let m = mat.nrows();
    let n = mat.ncols();

    let mut var = Col::<f32>::zeros(m);

    for j in 0..n {
        zipped!(&mut var, &mut counts, mean.as_ref(), mat.as_ref().col(j)).for_each(
            |unzipped!(mut var, mut counts, mean, mat)| {
                if !(*mat).is_nan() && !(*mean).is_nan() {
                    let x = *mat - *mean;
                    *var += x * x;
                    *counts += 1.0;
                }
            },
        );
    }

    zipped!(&mut var, counts.as_ref()).for_each(|unzipped!(mut var, counts)| {
        *var /= if *counts == 0.0 { 0.0 } else { *counts - 1.0 };
    });

    var
}

/// Compute the column variance, ignoring masked (NaN) values
pub fn column_var(mat: MatRef<'_, f32>) -> Col<f32> {
    let mut counts = Col::zeros(mat.nrows());
    let mean = column_mean_with_counts(mat, counts.as_mut());
    counts = Col::zeros(mat.nrows());
    column_var_with_mean_counts(mat, mean.as_ref(), counts.as_mut())
}

/// Compute the row mean, ignoring masked (NaN) values
#[inline(always)]
pub fn row_mean(mat: MatRef<'_, f32>) -> Row<f32> {
    // Create the type for the SIMD
    const N: usize = 8;
    type F32s = Simd<f32, N>;

    let _m = mat.nrows();
    let n = mat.ncols();

    let mut mean = Row::<f32>::zeros(n);

    for j in 0..n {
        // Get the column (slice of contiguous memory)
        let col = mat.col(j).try_as_slice().unwrap();

        // SIMD vectors we're summing into and using as "constants"
        let zeros = F32s::splat(0.0);
        let ones = F32s::splat(1.0);
        let mut vsum = F32s::splat(0.0);
        let mut vcount = F32s::splat(0.0);

        // Operate one SIMD vector-length chunk at a time
        let mut chunk_iter = col.chunks_exact(N);
        for chunk in chunk_iter.by_ref() {
            // Create the SIMD vector from this slice
            let v = F32s::from_slice(chunk);
            // Create the NaN mask
            let m = v.is_nan();
            // Count the NaNs
            vcount += m.select(zeros, ones);
            // Sum the masked non-nan values
            vsum += m.select(zeros, v);
        }

        // Sum all the result vector
        let mut count = vcount.reduce_sum();
        let mut sum = vsum.reduce_sum();

        // Handle the remaining bits that don't fit in a SIMD vector
        let tail = chunk_iter.remainder();
        if !tail.is_empty() {
            tail.iter().filter(|x| !x.is_nan()).for_each(|x| {
                sum += x;
                count += 1.0;
            })
        }

        // And assign to the mean
        mean[j] = sum / count;
    }

    mean
}

#[inline(always)]
fn row_var_with_mean(mat: MatRef<'_, f32>, mean: RowRef<'_, f32>) -> Row<f32> {
    const N: usize = 8;
    type F32s = Simd<f32, N>;

    let _m = mat.nrows();
    let n = mat.ncols();

    let mut var = Row::<f32>::zeros(n);

    for j in 0..n {
        let col = mat.col(j).try_as_slice().unwrap();
        let scalar_mean = mean[j];

        let means = F32s::splat(scalar_mean);
        let zeros = F32s::splat(0.0);
        let ones = F32s::splat(1.0);
        let mut vsum = F32s::splat(0.0);
        let mut vcount = F32s::splat(0.0);

        let mut chunk_iter = col.chunks_exact(N);
        for chunk in chunk_iter.by_ref() {
            let v = F32s::from_slice(chunk);
            let diff = v - means;
            let diff_squared = diff * diff;
            let m = diff_squared.is_nan();
            vcount += m.select(zeros, ones);
            vsum += m.select(zeros, v);
        }

        let mut count = vcount.reduce_sum();
        let mut sum = vsum.reduce_sum();

        let tail = chunk_iter.remainder();
        if !tail.is_empty() {
            for x in tail {
                let v = x - scalar_mean;
                let v2 = v * v;
                if !v2.is_nan() {
                    sum += v2;
                    count += 1.0;
                }
            }
        }

        var[j] = sum / if count == 0.0 { count } else { count - 1.0 };
    }

    var
}

/// Compute the row variance, ignoring masked (NaN) values
pub fn row_var(mat: MatRef<'_, f32>) -> Row<f32> {
    let mean = row_mean(mat);
    row_var_with_mean(mat, mean.as_ref())
}

#[cfg(test)]
pub mod test {
    use super::*;
    use faer::mat;
    use std::f32::NAN;

    #[test]
    fn test_col_mean() {
        let m = mat![
            [1.0, 2.0, NAN],
            [NAN, 5.0, 6.0],
            [7.0, NAN, 9.0],
            [NAN, 11.0, NAN]
        ];
        let mean = column_mean(m.as_ref());
        assert_eq!(mean.as_slice(), [1.5, 5.5, 8.0, 11.0].as_slice())
    }

    #[test]
    fn test_row_mean() {
        let m = mat![
            [1.0, 2.0, NAN],
            [NAN, 5.0, 6.0],
            [7.0, NAN, 9.0],
            [NAN, 11.0, NAN]
        ];
        let mean = row_mean(m.as_ref());
        assert_eq!(mean.as_slice(), [4.0, 6.0, 7.5].as_slice())
    }

    #[test]
    fn test_col_var() {
        let m = mat![
            [1.0, 2.0, NAN],
            [NAN, 5.0, 6.0],
            [7.0, NAN, 9.0],
            [NAN, NAN, NAN]
        ];
        let var = column_var(m.as_ref());
        assert_eq!(var[0], 0.5);
        assert_eq!(var[1], 0.5);
        assert_eq!(var[2], 2.0);
        assert!(var[3].is_nan());
    }

    #[test]
    fn test_col_var_bad_row() {
        let m = mat![
            [1.0, 2.0, NAN],
            [NAN, 5.0, 6.0],
            [7.0, NAN, 9.0],
            [NAN, NAN, NAN]
        ];
        let var = column_var(m.as_ref());
        assert_eq!(var[0], 0.5);
        assert_eq!(var[1], 0.5);
        assert_eq!(var[2], 2.0);
        assert!(var[3].is_nan());
    }

    #[test]
    fn test_col_var_bad_both() {
        let m = mat![
            [1.0, 2.0, NAN],
            [NAN, 5.0, NAN],
            [7.0, NAN, NAN],
            [NAN, NAN, NAN]
        ];
        let var = column_var(m.as_ref());
        assert_eq!(var[0], 0.5);
        assert!(var[1].is_nan());
        assert!(var[2].is_nan());
        assert!(var[3].is_nan());
    }

    #[test]
    fn test_row_var() {
        let m = mat![
            [1.0, 2.0, NAN],
            [NAN, 5.0, 6.0],
            [7.0, NAN, 9.0],
            [NAN, NAN, NAN]
        ];
        let var = row_var(m.as_ref());
        assert_eq!(var[0], 18.0);
        assert_eq!(var[1], 4.5);
        assert_eq!(var[2], 4.5);
    }

    #[test]
    fn test_row_var_bad_col() {
        let m = mat![
            [1.0, 2.0, NAN],
            [NAN, 5.0, NAN],
            [7.0, NAN, NAN],
            [NAN, 11.0, NAN]
        ];
        let var = row_var(m.as_ref());
        assert_eq!(var[0], 18.0);
        assert_eq!(var[1], 21.0);
        assert!(var[2].is_nan());
    }

    #[test]
    fn test_row_var_bad_both() {
        let m = mat![
            [1.0, 2.0, NAN],
            [NAN, 5.0, NAN],
            [7.0, NAN, NAN],
            [NAN, NAN, NAN]
        ];
        let var = row_var(m.as_ref());
        assert_eq!(var[0], 18.0);
        assert_eq!(var[1], 4.5);
        assert!(var[2].is_nan());
    }
}
