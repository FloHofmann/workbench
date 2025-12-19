//pub mod errors;
//pub mod types;

pub mod core;
pub mod py;

use pyo3::prelude::*;

#[pymodule]
fn spikeops(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    py::register(m)?;
    Ok(())
}
