use pyo3::prelude::*;

pub mod detect;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect::detect_peaks, m)?)?;
    Ok(())
}
