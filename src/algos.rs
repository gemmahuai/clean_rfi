use std::f32::NAN;

use crate::math::*;
use faer::{
    col,
    prelude::*,
    reborrow::{Reborrow, ReborrowMut},
};

/// Detrend variation across columns
pub fn detrend_columns(mut mat: MatMut<'_, f32>, order: usize) {
    let m = mat.nrows();
    let n = mat.ncols();

    // Compute the mean row
    let mean = row_mean(mat.rb());

    // Collect the x/y pairs that are *not* NaN
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for (j, x) in mean.as_slice().iter().enumerate() {
        if !x.is_nan() {
            ys.push(*x);
            xs.push(j as f32);
        }
    }
    // Put these ys into a column as that's what lstsq needs
    let ys: ColRef<'_, f32> = col::from_slice(&ys[..]);

    // Fit a polynomial to the mean row
    let mut v = vander(&xs, order);
    let coeffs = v.qr().solve_lstsq(ys);

    // Then make a new vandermonde to evaluate the polynomial with all x values
    // We need the full x range now
    xs = (0..n).map(|x| x as f32).collect();
    v = vander(&xs, order);
    let polyeval = v * coeffs;

    // Remove the low-order variation on the mean row across all columns
    for j in 0..n {
        let mut col = mat.rb_mut().col_mut(j);
        let weights = Col::from_fn(m, |_| polyeval[j]);
        col -= &weights;
    }
}

/// Detrend variation across rows.
pub fn detrend_rows(mut mat: MatMut<'_, f32>, order: usize) {
    let m = mat.nrows();
    let n = mat.ncols();

    // Compute the mean column
    let mean = column_mean(mat.rb());

    // Collect the x/y pairs that are *not* NaN
    let mut xs = Vec::with_capacity(m);
    let mut ys = Vec::with_capacity(m);
    for (i, x) in mean.as_slice().iter().enumerate() {
        if !x.is_nan() {
            ys.push(*x);
            xs.push(i as f32);
        }
    }
    // Put these ys into a column as that's what lstsq needs
    let ys: ColRef<'_, f32> = col::from_slice(&ys[..]);

    // Fit a polynomial to the mean column
    let mut v = vander(&xs, order);
    let coeffs = v.qr().solve_lstsq(ys);

    // Then make a new vandermonde to evaluate the polynomial with all x values
    // We need the full x range now
    xs = (0..m).map(|x| x as f32).collect();
    v = vander(&xs, order);
    let polyeval = v * coeffs;

    // Remove the low-order variation on the mean column across all rows
    for j in 0..n {
        let mut col = mat.rb_mut().col_mut(j);
        col -= &polyeval;
    }
}

/// Mask off channels just given their channel number
pub fn channel_mask(mat: MatMut<'_, f32>, channels: &[usize]) {
    // Channels are rows here (column-major)
    let mask = Col::<f32>::from_fn(
        mat.nrows(),
        |i| {
            if channels.contains(&i) {
                NAN
            } else {
                0.0
            }
        },
    );
    mask_rows(mat, mask.as_ref());
}

/// Mask off time samples (columns) whose mean standard dev (in frequency) is above some threshold
pub fn varcut_time(mat: MatMut<'_, f32>, sigma_threshold: f32) {
    let n = mat.ncols();

    // Compute the standard deviations
    let mut sigma = row_var(mat.rb());
    zipped!(&mut sigma).for_each(|unzipped!(mut x)| *x = (*x).sqrt());

    // Compute the mean of the standard deviations
    let mu_sigma = simd_mean(sigma.as_slice());

    // Compute the standard deviation of the standard deviations
    let sigma_sigma = simd_var_with_mean(sigma.as_slice(), mu_sigma).sqrt();

    // Compute the mask where sigma > mu_sigma + sigma_threshold * sigma_sigma
    let mut mask = Row::<f32>::zeros(n);
    zipped!(&mut mask, &sigma).for_each(|unzipped!(mut mask, sigma)| {
        if *sigma > mu_sigma + sigma_threshold * sigma_sigma {
            *mask = NAN;
        }
    });

    // And assign NaNs to those columns
    mask_columns(mat, mask.as_ref());
}

/// Mask off channels (rows) whose mean standard dev (in time) is above some threshold
pub fn varcut_channels(mat: MatMut<'_, f32>, threshold: f32) {
    let m = mat.nrows();

    // Compute the standard deviations
    let mut sigma = column_var(mat.rb());
    zipped!(&mut sigma).for_each(|unzipped!(mut x)| *x = (*x).sqrt());

    // Compute the mean of the standard deviations
    let mu_sigma = simd_mean(sigma.as_slice());

    // Compute the standard deviation of the standard deviations
    let sigma_sigma = simd_var_with_mean(sigma.as_slice(), mu_sigma).sqrt();

    // Compute the mask where sigma > mu_sigma + sigma_threshold * sigma_sigma
    let mut mask = Col::<f32>::zeros(m);
    zipped!(&mut mask, &sigma).for_each(|unzipped!(mut mask, sigma)| {
        if *sigma > mu_sigma + threshold * sigma_sigma {
            *mask = NAN;
        }
    });

    // And assign NaNs to those columns
    mask_rows(mat, mask.as_ref());
}

/// Normalize bandpass and remove dead channels (channels with low power)
pub fn normalize_and_trim_bandpass(mut mat: MatMut<'_, f32>, threshold: f32) {
    let n = mat.ncols();
    let m = mat.nrows();
    // Compute bandpaass
    let t_sys = column_mean(mat.as_ref());
    // Compute the median
    let t_sys_median = median(t_sys.as_slice());
    // Find bad channels
    let mut mask = Col::<f32>::zeros(m);
    zipped!(&mut mask, &t_sys).for_each(|unzipped!(mut mask, t_sys)| {
        if *t_sys < threshold * t_sys_median {
            *mask = NAN;
        }
    });
    // Apply mask
    mask_rows(mat.rb_mut(), mask.as_ref());
    // Normalize bandpass
    for j in 0..n {
        zipped!(mat.as_mut().col_mut(j), t_sys.as_ref()).for_each(|unzipped!(mut mat, t_sys)| {
            *mat /= *t_sys;
        });
    }
}

/// Mask time samples that contain "DM=0" outliers
pub fn dm_zero_filter(mut mat: MatMut<'_, f32>, threshold: f32) {
    let n = mat.ncols();
    // Find the average time series (sum all frequencies)
    let mut dm_zero = row_mean(mat.as_ref());
    let dm_zero_median = median(dm_zero.as_slice());

    // Subtract out the median in place
    zipped!(&mut dm_zero).for_each(|unzipped!(mut x)| *x -= dm_zero_median);

    // Compute the mean absolute deviation
    let dm_zero_mad = simd_mad(dm_zero.as_slice());
    let std_dev = 1.4826 * dm_zero_mad;

    // Find the outliers
    let mut mask = Row::<f32>::zeros(n);
    zipped!(&mut mask, dm_zero).for_each(|unzipped!(mut mask, dm_zero)| {
        if (*dm_zero).abs() > threshold * std_dev {
            *mask = NAN;
        }
    });

    // Apply mask
    mask_columns(mat.rb_mut(), mask.as_ref());
}

/// Clean a block of time/freq data, used in every IO operation
/// This is the top-level cleaning function
pub fn clean_block(mut mat: MatMut<'_, f32>) {
    // Start with a dumb mask of the top and bottom frequency channels
    // Those are full of nonsense all the time
    let mut bad_channels = vec![];
    bad_channels.append(&mut (0..=250).collect());
    bad_channels.append(&mut (1797..=2047).collect());
    channel_mask(mat.rb_mut(), &bad_channels);

    // Remove bandpass variation
    normalize_and_trim_bandpass(mat.rb_mut(), 0.001);

    // Twice-iterative variance cut across both axes
    varcut_channels(mat.rb_mut(), 3.);
    varcut_time(mat.rb_mut(), 5.);
    varcut_channels(mat.rb_mut(), 5.);
    varcut_time(mat.rb_mut(), 7.);

    // Mask out the DM=0 values
    //dm_zero_filter(mat.rb_mut(), 7.0);

    // Remove variation across frequency and time
    detrend_rows(mat.rb_mut(), 4);
    detrend_columns(mat, 4);
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use faer::{assert_matrix_eq, mat};

    #[test]
    fn test_detrend_freq() {
        let mut a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        detrend_columns(a.as_mut(), 2);
        let detrended = mat![[-3.0f32, -3.0, -3.0], [0.0, 0.0, 0.0], [3.0, 3.0, 3.0]];
        assert_matrix_eq!(a, detrended, comp = abs, tol = 1e-6);
    }

    #[test]
    fn test_detrend_time() {
        let mut a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        detrend_rows(a.as_mut(), 2);
        let detrended = mat![[-1., 0., 1.], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0f32]];
        assert_matrix_eq!(a, detrended, comp = abs, tol = 1e-6);
    }
}
