// Conventions:
// indexing is always row,column
// which will be represented in i,j
// the length of these is m,n such that 0 <= i < m and 0 <= j < n
// Faer mats are column-major, so in-order indexing is idx = i + jm

// Arrays are masked with NaNs and unmasked with 0

use faer::prelude::*;

pub mod mask;
pub mod mean;
pub mod var;

/// Construct the vandermonde matrix for x values `xs`
pub fn vander(xs: &[f32], order: usize) -> Mat<f32> {
    Mat::from_fn(xs.len(), order + 1, |i, j| xs[i].powf(j as f32))
}

/// Compute the median of a slice (ignoring NaNs)
pub fn median(xs: &[f32]) -> f32 {
    let mut non_nans: Vec<_> = xs.iter().filter(|x| !x.is_nan()).collect();
    non_nans.sort_by(|x, y| x.total_cmp(y));
    let mid = non_nans.len() / 2;
    *non_nans[mid]
}

#[cfg(test)]
pub mod test {
    use super::*;

    const NAN: f32 = f32::NAN;

    #[test]
    fn test_vander() {
        let xs = vander([1.0, 2.0, 3.0].as_slice(), 2);
        assert_eq!(xs.col_as_slice(0), [1.0, 1.0, 1.0].as_slice());
        assert_eq!(xs.col_as_slice(1), [1.0, 2.0, 3.0].as_slice());
        assert_eq!(xs.col_as_slice(2), [1.0, 4.0, 9.0].as_slice());
    }

    #[test]
    fn test_median() {
        let xs = vec![5., 1., NAN, 4., 2., 3.];
        let med = median(xs.as_slice());
        assert_eq!(med, 3.0);
    }
}
