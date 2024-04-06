use crate::math::*;
use faer::{
    prelude::*,
    reborrow::{Reborrow, ReborrowMut},
};

/// Detrend variation across columns
pub fn detrend_columns_inplace(mut data_view: MatMut<'_, f32>, order: usize) {
    let n_cols = data_view.ncols();
    let n_rows = data_view.nrows();
    let xs: Vec<_> = (0..n_cols).map(|x| x as f32).collect();
    // Compute the mean row
    let ys = row_mean(data_view.rb());
    // Fit a polynomial to the mean row and evaluate that polynomial
    let v = vander(&xs, order);
    let coeffs = v.qr().solve_lstsq(ys.transpose());
    let polyeval = v * coeffs;
    // Remove the low-order variation on the mean row across all columns
    for j in 0..data_view.ncols() {
        let mut col = data_view.rb_mut().col_mut(j);
        let weights = Col::from_fn(n_rows, |_| polyeval[j]);
        col -= &weights;
    }
}

/// Detrend variation across rows
pub fn detrend_rows_inplace(mut data_view: MatMut<'_, f32>, order: usize) {
    let n_cols = data_view.ncols();
    let n_rows = data_view.nrows();
    let xs: Vec<_> = (0..n_rows).map(|x| x as f32).collect();
    // Compute the mean column
    let ys = column_mean(data_view.rb());
    // Fit a polynomial to the mean column and evaluate that polynomial
    let v = vander(&xs, order);
    let coeffs = v.qr().solve_lstsq(ys);
    // Then use the vandermonde matrix to evaluate the polynomial
    let polyeval = v * coeffs;
    // Remove the low-order variation on the mean column across all rows
    for j in 0..n_cols {
        let mut col = data_view.rb_mut().col_mut(j);
        col -= &polyeval;
    }
}

/// Clean a block of time/freq data, used in every IO operation
/// This is the top-level cleaning function
pub fn clean_block(mut data_view: MatMut<'_, f32>) {
    // Remove variation across frequencies
    detrend_rows_inplace(data_view.rb_mut(), 4);
    // Then remove variation across time
    detrend_columns_inplace(data_view, 4);
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use faer::{assert_matrix_eq, mat};

    #[test]
    fn test_detrend_freq() {
        let mut a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        detrend_columns_inplace(a.as_mut(), 2);
        let detrended = mat![[-3.0f32, -3.0, -3.0], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]];
        assert_matrix_eq!(a, detrended, comp = abs, tol = 1e-6);
    }

    #[test]
    fn test_detrend_time() {
        let mut a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        detrend_rows_inplace(a.as_mut(), 2);
        let detrended = mat![[-1., 0., 1.], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0f32]];
        assert_matrix_eq!(a, detrended, comp = abs, tol = 1e-6);
    }
}
