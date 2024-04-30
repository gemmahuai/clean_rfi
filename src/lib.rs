pub mod algos;
pub mod math;

#[cfg(feature = "cli")]
pub mod io;

#[cfg(feature = "python")]
pub mod python {
    use super::*;

    use faer_ext::IntoFaer;
    use numpy::{ndarray::ArrayViewMut2, PyArray2, PyArrayMethods};
    use pyo3::prelude::*;

    fn clean_block(
        mat: ArrayViewMut2<'_, f32>,
        first_pass_sigma: f32,
        second_pass_sigma: f32,
        detrend_order: usize,
        include_detrending: bool,
    ) {
        // Transpose so the Mat is column-major when we hand it to faer
        let faer_block = mat.into_faer().transpose_mut();
        algos::clean_block(
            faer_block,
            first_pass_sigma,
            second_pass_sigma,
            detrend_order,
            include_detrending,
        );
    }

    #[pyfunction]
    #[pyo3(name = "clean_block", signature = (mat, first_pass_sigma=3.0, second_pass_sigma=5.0, detrend_order=4))]
    fn clean_block_py(
        mat: &Bound<'_, PyArray2<f32>>,
        first_pass_sigma: f32,
        second_pass_sigma: f32,
        detrend_order: usize,
        include_detrending: bool,
    ) {
        let mat = unsafe { mat.as_array_mut() };
        clean_block(mat, first_pass_sigma, second_pass_sigma, detrend_order, include_detrending);
    }

    #[pymodule]
    fn clean_rfi<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(clean_block_py, m)?)?;
        Ok(())
    }
}
