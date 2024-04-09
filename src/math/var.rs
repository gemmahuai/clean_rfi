use crate::math::mean::column_mean;
use faer::prelude::*;
use nanstats::NaNVar;
use pulp::Simd;

#[pulp::with_simd(col_var_col_major_with_mean = pulp::Arch::new())]
#[inline(always)]
fn col_var_col_major_with_mean_simd<S: Simd>(
    simd: S,
    mat: MatRef<'_, f32>,
    mean: ColRef<'_, f32>,
) -> Col<f32> {
    _ = simd;
    let m = mat.nrows();
    let n = mat.ncols();
    let mut var = Col::<f32>::zeros(m);
    let mut counts = Col::<f32>::zeros(m);
    for j in 0..n {
        zipped!(&mut var, &mut counts, mean.as_ref(), mat.as_ref().col(j)).for_each(
            |unzipped!(mut var, mut counts, mean, mat)| {
                if !(*mat).is_nan() {
                    let x = *mat - *mean;
                    *var += x * x;
                    *counts += 1.0;
                }
            },
        );
    }
    zipped!(&mut var, counts.as_ref()).for_each(|unzipped!(mut var, counts)| {
        *var /= if *counts == 0.0 {
            f32::NAN
        } else {
            *counts - 1.0
        };
    });
    var
}

fn col_var_row_major_with_mean(mat: MatRef<'_, f32>, mean: ColRef<'_, f32>) -> Col<f32> {
    let m = mat.nrows();
    let mut var = Col::<f32>::zeros(m);
    for i in 0..m {
        let row = mat.row(i).try_as_slice().unwrap();
        let scalar_mean = mean[i];
        var[i] = row.nanvar_with_mean(scalar_mean);
    }
    var
}

/// Compute the column var, ignoring masked (NaN) values
pub fn column_var(mat: MatRef<'_, f32>) -> Col<f32> {
    let mean = column_mean(mat);
    if mat.col_stride() == 1 {
        col_var_row_major_with_mean(mat, mean.as_ref())
    } else {
        col_var_col_major_with_mean(mat, mean.as_ref())
    }
}

/// Compute the row variance, ignoring masked (NaN) values
pub fn row_var(mat: MatRef<'_, f32>) -> Row<f32> {
    column_var(mat.transpose()).transpose().to_owned()
}

#[cfg(test)]
pub mod test {
    use super::*;
    use faer::mat;

    const NAN: f32 = f32::NAN;

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
