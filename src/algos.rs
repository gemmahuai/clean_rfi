use crate::math::{
    mask::{mask_columns, mask_rows},
    mean::{column_mean, row_mean},
    median, vander,
    var::{column_var, row_var},
};
use faer::{
    prelude::*,
    reborrow::{Reborrow, ReborrowMut},
};
use nanstats::*;
use pulp::Simd;

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

    // If it was all NaNs, nothing we can do...
    if ys.is_empty() {
        return;
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

    // If it was all NaNs, nothing we can do...
    if ys.is_empty() {
        return;
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
    let mask = Col::<f32>::from_fn(mat.nrows(), |i| {
        if channels.contains(&i) {
            f32::NAN
        } else {
            0.0
        }
    });
    mask_rows(mat, mask.as_ref());
}

/// Mask off time samples (columns) whose mean standard dev (in frequency) is above some threshold
#[pulp::with_simd(varcut_time = pulp::Arch::new())]
#[inline(always)]
pub fn varcut_time_with_simd<S: Simd>(simd: S, mat: MatMut<'_, f32>, sigma_threshold: f32) {
    _ = simd;
    let n = mat.ncols();

    // Compute the standard deviations
    let mut sigma = row_var(mat.rb());
    zipped!(&mut sigma).for_each(|unzipped!(mut x)| *x = (*x).sqrt());

    // Compute the mean of the standard deviations
    let mu_sigma = sigma.as_slice().nanmean();

    // Compute the standard deviation of the standard deviations
    let sigma_sigma = sigma.as_slice().nanvar_with_mean(mu_sigma).sqrt();

    // Compute the mask where sigma > mu_sigma + sigma_threshold * sigma_sigma
    let mut mask = Row::<f32>::zeros(n);
    zipped!(&mut mask, &sigma).for_each(|unzipped!(mut mask, sigma)| {
        if *sigma > mu_sigma + sigma_threshold * sigma_sigma {
            *mask = f32::NAN;
        }
    });

    // And assign NaNs to those columns
    mask_columns(mat, mask.as_ref());
}

/// Mask off channels (rows) whose mean standard dev (in time) is above some threshold
#[pulp::with_simd(varcut_channels = pulp::Arch::new())]
#[inline(always)]
pub fn varcut_channels_with_simd<S: Simd>(simd: S, mat: MatMut<'_, f32>, threshold: f32) {
    _ = simd;
    let m = mat.nrows();

    // Compute the standard deviations
    let mut sigma = column_var(mat.rb());
    zipped!(&mut sigma).for_each(|unzipped!(mut x)| *x = (*x).sqrt());

    // Compute the mean of the standard deviations
    let mu_sigma = sigma.as_slice().nanmean();

    // Compute the standard deviation of the standard deviations
    let sigma_sigma = sigma.as_slice().nanvar_with_mean(mu_sigma).sqrt();

    // Compute the mask where sigma > mu_sigma + sigma_threshold * sigma_sigma
    let mut mask = Col::<f32>::zeros(m);
    zipped!(&mut mask, &sigma).for_each(|unzipped!(mut mask, sigma)| {
        if *sigma > mu_sigma + threshold * sigma_sigma {
            *mask = f32::NAN;
        }
    });

    // And assign NaNs to those columns
    mask_rows(mat, mask.as_ref());
}

/// Normalize bandpass and remove dead channels (channels with low power)
#[pulp::with_simd(normalize_and_trim_bandpass = pulp::Arch::new())]
#[inline(always)]
pub fn normalize_and_trim_bandpass_with_simd<S: Simd>(
    simd: S,
    mut mat: MatMut<'_, f32>,
    threshold: f32,
) {
    _ = simd;
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
            *mask = f32::NAN;
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

/// Clean a block of time/freq data, used in every IO operation
/// This is the top-level cleaning function
pub fn clean_block(
    mut mat: MatMut<'_, f32>,
    first_pass_sigma: f32,
    second_pass_sigma: f32,
    detrend_order: usize,
) {
    // Start with a dumb mask of the top and bottom frequency channels
    // Those are full of nonsense all the time
    let mut bad_channels = vec![];
    bad_channels.append(&mut (0..=250).collect());
    bad_channels.append(&mut (1797..=2047).collect());
    channel_mask(mat.rb_mut(), &bad_channels);

    // Remove bandpass variation
    normalize_and_trim_bandpass(mat.rb_mut(), 0.001);

    // Twice-iterative variance cut across both axes
    varcut_channels(mat.rb_mut(), first_pass_sigma);
    varcut_time(mat.rb_mut(), first_pass_sigma);
    varcut_channels(mat.rb_mut(), second_pass_sigma);
    varcut_time(mat.rb_mut(), second_pass_sigma);

    // Remove variation across frequency and time
    detrend_rows(mat.rb_mut(), detrend_order);
    detrend_columns(mat, detrend_order);
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
