// #[pyfunction]wrappers for detect
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::core;

/// Python detect_peaks(trace: np.ndarray, threshold: float, min_distance: int -> np.ndarray[int64]
#[pyfunction]
pub fn detect_peaks<'py>(
    py: Python<'py>,
    trace: PyReadonlyArray1<'py, f64>,
    threshold: f64,
    upper_threshold: f64,
    min_distance: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let trace_slice = trace.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("trace must be a contiguous 1D float64 array")
    })?;
    let peaks = core::detect::detect_peaks(trace_slice, threshold, upper_threshold, min_distance);

    // Convert usize -> i64 for a nice NUmPy dtype on the Python side
    let out: Vec<i64> = peaks.into_iter().map(|i| i as i64).collect();
    Ok(PyArray1::from_vec(py, out))
}
