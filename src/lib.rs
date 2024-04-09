pub mod algos;
pub mod io;
pub mod math;

#[cfg(feature = "python")]
pub mod python {
    use super::*;

    use faer_ext::IntoFaer;
    use numpy::{ndarray::ArrayViewMut2, PyArray2, PyArrayMethods};
    use pyo3::prelude::*;

    #[pymodule]
    fn clean_rfi<'py>(_py: Python<'py>, m: &Bound<'py, PyModule>) -> PyResult<()> {
        fn clean_block(mat: ArrayViewMut2<'_, f32>) {
            // Transpose so the Mat is column-major when we hand it to faer
            let faer_block = mat.into_faer().transpose_mut();
            algos::clean_block(faer_block);
        }

        #[pyfn(m)]
        #[pyo3(name = "clean_block")]
        fn clean_block_py(mat: &Bound<'_, PyArray2<f32>>) {
            let mat = unsafe { mat.as_array_mut() };
            clean_block(mat);
        }

        Ok(())
    }
}
