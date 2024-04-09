use faer::prelude::*;
use nanstats::NaNMean;
use pulp::Simd;

#[pulp::with_simd(col_mean_col_major = pulp::Arch::new())]
#[inline(always)]
fn col_mean_col_major_with_simd<S: Simd>(simd: S, mat: MatRef<'_, f32>) -> Col<f32> {
    _ = simd;
    let m = mat.nrows();
    let n = mat.ncols();
    let mut counts = Col::<f32>::zeros(m);
    let mut mean = Col::<f32>::zeros(m);
    for j in 0..n {
        zipped!(&mut mean, &mut counts, mat.as_ref().col(j)).for_each(
            |unzipped!(mut mean, mut counts, data)| {
                if !(*data).is_nan() {
                    *mean += *data;
                    *counts += 1.;
                }
            },
        );
    }
    zipped!(&mut mean, counts.as_ref()).for_each(|unzipped!(mut mean, counts)| {
        *mean /= *counts;
    });
    mean
}

/// Compute the row mean, ignoring masked (NaN) values
fn col_mean_row_major(mat: MatRef<'_, f32>) -> Col<f32> {
    let m = mat.nrows();
    let mut mean = Col::<f32>::zeros(m);
    for i in 0..m {
        // Get the row (slice of contiguous memory)
        let row = mat.row(i).try_as_slice().unwrap();
        mean[i] = row.nanmean();
    }
    mean
}

/// Compute the column mean, ignoring masked (NaN) values
pub fn column_mean(mat: MatRef<'_, f32>) -> Col<f32> {
    if mat.col_stride() == 1 {
        col_mean_row_major(mat)
    } else {
        col_mean_col_major(mat)
    }
}

/// Compute the row mean, ignoring masked (NaN) values
pub fn row_mean(mat: MatRef<'_, f32>) -> Row<f32> {
    column_mean(mat.transpose()).transpose().to_owned()
}

#[cfg(test)]
pub mod test {
    use super::*;
    use faer::mat;

    const NAN: f32 = f32::NAN;

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
}
