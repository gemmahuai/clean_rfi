use faer::prelude::*;

// --------------------------- TO BE INCLUDED IN FAER EVENTUALLY

fn col_mean_col_major(mat: MatRef<'_, f32>) -> Col<f32> {
    struct Impl<'a> {
        mat: MatRef<'a, f32>,
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = Col<f32>;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { mat } = self;
            _ = simd;

            let m = mat.nrows();
            let n = mat.ncols();
            let one_n = 1.0 / n as f32;

            let mut mean = Col::<f32>::zeros(m);
            for j in 0..n {
                zipped!(&mut mean, mat.col(j)).for_each(|unzipped!(mut mean, mat)| {
                    *mean += *mat * one_n;
                });
            }
            mean
        }
    }

    pulp::Arch::new().dispatch(Impl { mat })
}

fn col_meanvar_col_major(mat: MatRef<'_, f32>) -> (Col<f32>, Col<f32>) {
    struct Impl<'a> {
        mat: MatRef<'a, f32>,
        mean: Col<f32>,
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = (Col<f32>, Col<f32>);

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { mat, mean } = self;
            _ = simd;
            let m = mat.nrows();
            let n = mat.ncols();
            let one_n1 = 1.0 / (n - 1) as f32;

            let mut variance: Col<f32> = Col::zeros(m);
            for i in 0..n {
                zipped!(&mut variance, &mean, mat.col(i)).for_each(
                    |unzipped!(mut var, mean, mat)| {
                        let x = *mat - *mean;
                        *var += x * x * one_n1;
                    },
                );
            }
            (mean, variance)
        }
    }

    let mean = col_mean_col_major(mat);
    pulp::Arch::new().dispatch(Impl { mat, mean })
}

fn col_mean_row_major(mat: MatRef<'_, f32>) -> Col<f32> {
    struct Impl<'a> {
        mat: MatRef<'a, f32>,
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = Col<f32>;

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { mat } = self;

            let m = mat.nrows();
            let n = mat.ncols();
            let one_n = 1.0 / n as f32;

            let mut mean = Col::<f32>::zeros(m);
            for i in 0..m {
                let mut sum0 = simd.f32s_splat(0.0);
                let mut sum1 = simd.f32s_splat(0.0);
                let mut sum2 = simd.f32s_splat(0.0);
                let mut sum3 = simd.f32s_splat(0.0);

                let row = mat.row(i).try_as_slice().unwrap();
                let (head, tail) = S::f32s_as_simd(row);
                let (head4, head1) = pulp::as_arrays::<4, _>(head);

                for &[x0, x1, x2, x3] in head4 {
                    sum0 = simd.f32s_add(sum0, x0);
                    sum1 = simd.f32s_add(sum1, x1);
                    sum2 = simd.f32s_add(sum2, x2);
                    sum3 = simd.f32s_add(sum3, x3);
                }
                for &x0 in head1 {
                    sum0 = simd.f32s_add(sum0, x0);
                }

                let sum0 = simd.f32s_add(sum0, sum1);
                let sum2 = simd.f32s_add(sum2, sum3);

                let sum0 = simd.f32s_add(sum0, sum2);

                let sum = simd.f32s_reduce_sum(sum0) + tail.iter().sum::<f32>();

                mean[i] = sum * one_n;
            }
            mean
        }
    }

    pulp::Arch::new().dispatch(Impl { mat })
}

fn col_meanvar_row_major(mat: MatRef<'_, f32>) -> (Col<f32>, Col<f32>) {
    struct Impl<'a> {
        mat: MatRef<'a, f32>,
        mean: Col<f32>,
    }

    impl pulp::WithSimd for Impl<'_> {
        type Output = (Col<f32>, Col<f32>);

        #[inline(always)]
        fn with_simd<S: pulp::Simd>(self, simd: S) -> Self::Output {
            let Self { mat, mean } = self;
            _ = simd;
            let m = mat.nrows();
            let n = mat.ncols();
            let one_n1 = 1.0 / (n - 1) as f32;

            let mut variance: Col<f32> = Col::zeros(m);

            for i in 0..m {
                let scalar_mean = mean[i];
                let mean = simd.f32s_splat(scalar_mean);

                let mut sum0 = simd.f32s_splat(0.0);
                let mut sum1 = simd.f32s_splat(0.0);
                let mut sum2 = simd.f32s_splat(0.0);
                let mut sum3 = simd.f32s_splat(0.0);

                let row = mat.row(i).try_as_slice().unwrap();
                let (head, tail) = S::f32s_as_simd(row);
                let (head4, head1) = pulp::as_arrays::<4, _>(head);

                for &[x0, x1, x2, x3] in head4 {
                    let x0 = simd.f32s_sub(x0, mean);
                    let x1 = simd.f32s_sub(x1, mean);
                    let x2 = simd.f32s_sub(x2, mean);
                    let x3 = simd.f32s_sub(x3, mean);

                    sum0 = simd.f32s_mul_add(x0, x0, sum0);
                    sum1 = simd.f32s_mul_add(x1, x1, sum1);
                    sum2 = simd.f32s_mul_add(x2, x2, sum2);
                    sum3 = simd.f32s_mul_add(x3, x3, sum3);
                }
                for &x0 in head1 {
                    let x0 = simd.f32s_sub(x0, mean);

                    sum0 = simd.f32s_mul_add(x0, x0, sum0);
                }

                let sum0 = simd.f32s_add(sum0, sum1);
                let sum2 = simd.f32s_add(sum2, sum3);

                let sum0 = simd.f32s_add(sum0, sum2);

                let sum = simd.f32s_reduce_sum(sum0)
                    + tail
                        .iter()
                        .map(|x| {
                            let var = x - scalar_mean;
                            var * var
                        })
                        .sum::<f32>();

                variance[i] = sum * one_n1;
            }

            (mean, variance)
        }
    }

    let mean = col_mean_row_major(mat);
    pulp::Arch::new().dispatch(Impl { mat, mean })
}

pub fn column_mean(mat: MatRef<'_, f32>) -> Col<f32> {
    if mat.col_stride() == 1 {
        col_mean_row_major(mat)
    } else {
        col_mean_col_major(mat)
    }
}

pub fn column_meanvar(mat: MatRef<'_, f32>) -> (Col<f32>, Col<f32>) {
    if mat.col_stride() == 1 {
        col_meanvar_row_major(mat)
    } else {
        col_meanvar_col_major(mat)
    }
}

pub fn row_mean(mat: MatRef<'_, f32>) -> Row<f32> {
    column_mean(mat.transpose()).transpose().to_owned()
}

pub fn row_meanvar(mat: MatRef<'_, f32>) -> (Row<f32>, Row<f32>) {
    let (mean, var) = column_meanvar(mat.transpose());
    (mean.transpose().to_owned(), var.transpose().to_owned())
}

// --------------------------- END

/// Construct the vandermonde matrix for x values `xs`
pub fn vander(xs: &[f32], order: usize) -> Mat<f32> {
    Mat::from_fn(xs.len(), order + 1, |i, j| xs[i].powf(j as f32))
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn test_vander() {
        let xs = vander([1.0, 2.0, 3.0].as_slice(), 2);
        assert_eq!(xs.col_as_slice(0), [1.0, 1.0, 1.0].as_slice());
        assert_eq!(xs.col_as_slice(1), [1.0, 2.0, 3.0].as_slice());
        assert_eq!(xs.col_as_slice(2), [1.0, 4.0, 9.0].as_slice());
    }

    #[test]
    fn test_col_mean() {
        let a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0f32]];
        let mean = column_mean(a.as_ref());
        assert_eq!(mean.as_slice(), [2.0, 5.0, 8.0].as_slice());
    }

    #[test]
    fn test_row_mean() {
        let a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0f32]];
        let mean = row_mean(a.as_ref());
        assert_eq!(mean.as_slice(), [4.0, 5.0, 6.0].as_slice());
    }

    #[test]
    fn test_col_meanvar() {
        let a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0f32]];
        let (mean, var) = column_meanvar(a.as_ref());
        assert_eq!(mean.as_slice(), [2.0, 5.0, 8.0].as_slice());
        assert_eq!(var.as_slice(), [1.0, 1.0, 1.0].as_slice())
    }

    #[test]
    fn test_row_meanvar() {
        let a = mat![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0f32]];
        let (mean, var) = row_meanvar(a.as_ref());
        assert_eq!(mean.as_slice(), [4.0, 5.0, 6.0].as_slice());
        assert_eq!(var.as_slice(), [9.0, 9.0, 9.0].as_slice())
    }
}
