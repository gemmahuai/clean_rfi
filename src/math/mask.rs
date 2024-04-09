use faer::prelude::*;
use pulp::Simd;

/// Given a matrix and a single row mask (NaN values and zeros), apply the mask to every row in the matrix
#[pulp::with_simd(mask_columns = pulp::Arch::new())]
#[inline(always)]
pub fn mask_columns_with_simd<S: Simd>(simd: S, mut mat: MatMut<'_, f32>, mask: RowRef<'_, f32>) {
    _ = simd;
    let n = mat.ncols();
    for j in 0..n {
        let mask_val = mask[j];
        zipped!(mat.as_mut().col_mut(j)).for_each(|unzipped!(mut mat)| {
            *mat += mask_val;
        });
    }
}

/// Given a matrix and a single column mask (NaN values and zeros), apply the mask to every column in the matrix
#[pulp::with_simd(mask_rows = pulp::Arch::new())]
#[inline(always)]
pub fn mask_rows_with_simd<S: Simd>(simd: S, mut mat: MatMut<'_, f32>, mask: ColRef<'_, f32>) {
    _ = simd;
    let n = mat.ncols();
    for j in 0..n {
        zipped!(mat.as_mut().col_mut(j), &mask).for_each(|unzipped!(mut mat, mask)| {
            *mat += *mask;
        });
    }
}

#[cfg(test)]
pub mod test {
    use super::*;
    use faer::{col, mat, row};

    const NAN: f32 = f32::NAN;

    #[test]
    fn test_column_mask() {
        let mut m = mat![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let mask = row![NAN, 0.0, 0.0];
        mask_columns(m.as_mut(), mask.as_ref());
        // TODO: Actually test the result
    }

    #[test]
    fn test_row_mask() {
        let mut m = mat![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let mask = col![0.0, NAN, 0.0, 0.0];
        mask_rows(m.as_mut(), mask.as_ref());
        // TODO: Actually test the result
    }
}
