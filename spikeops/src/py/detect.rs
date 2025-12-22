// #[pyfunction]wrappers for detect
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::core;

#[pyfunction]
/// detect_peaks of a filtered voltage trace
///
/// # Parameters
///
/// - 'trace': A ndarray of type float of filtered voltage trace
/// - 'threshold': float - sets the lower threshold for peak detection
/// - 'upper_threshold': float - sets the upper threshold for peak detection
/// - 'min_distance': int - the refractory period in samples to take into consideration
///
/// # Returns
///
/// A ndarray of type int where each entry refers to a sample a peak of a spike is detected on.
///
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

#[pyfunction]
///
pub fn extract_waveforms<'py>(
    py: Python<'py>,
    trace: PyReadonlyArray1<'py, f64>,
    spikesamples: PyReadonlyArray1<'py, i64>,
    sampling_rate: usize,
    time_pre: f64,
    time_post: f64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray2<f64>>)> {
    let trace_slice = trace.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("trace must be a contiguous 1D float64 array")
    })?;
    let spikesamples_i64 = spikesamples.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(
            "spikesamples must be a contiguous 1D array of ints",
        )
    })?;

    // convert spike indices ot usize safely
    let mut spikesamples_usize = Vec::with_capacity(spikesamples_i64.len());
    for &s in spikesamples_i64 {
        if s < 0 {
            continue; // ignore negative indices
        }
        let u = s as usize;
        if u < trace_slice.len() {
            spikesamples_usize.push(u);
        }
    }

    let (waveshapes, samples_per_wave, kept) = core::detect::extract_waveforms(
        trace_slice,
        &spikesamples_usize,
        sampling_rate,
        time_pre,
        time_post,
    );

    let kept_i64: Vec<i64> = kept.into_iter().map(|i| i as i64).collect();
    let kept_arr = PyArray1::from_vec(py, kept_i64);

    let wave_len = samples_per_wave;
    let n_waves = if wave_len == 0 {
        0
    } else {
        waveshapes.len() / wave_len
    };

    let arr = Array2::from_shape_vec((n_waves, wave_len), waveshapes).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "failed to reshape output (shape=({}, {})): {}",
            n_waves, wave_len, e
        ))
    })?;
    let waves_arr = arr.into_pyarray(py).to_owned();
    Ok((kept_arr, waves_arr))
}
