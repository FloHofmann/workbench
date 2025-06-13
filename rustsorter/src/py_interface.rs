use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAny, PyDict};
use std::path::Path;

pub fn retrieve_py_data(h5path: &Path) -> PyResult<Py<PyAny>> {
    Python::with_gil(|py| {
        let filepath = h5path.to_str().expect("non-UTF8 path");
        // Add workbench dir to sys.path
        py.import("sys")?
            .getattr("path")?
            .call_method1("insert", (0, "../../workbench"))?;

        // Import the python module and class
        let module = py.import("h5data.py")?;
        let h5data_class = module.getattr("h5data")?;

        let pathlib = py.import("pathlib")?;
        let path_obj = pathlib.call_method1("Path", (filepath,))?;

        // Call the Python class constructor
        let instance = h5data_class.call1((path_obj,))?;

        Ok(instance.into())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyDict;
    use std::path::Path;

    #[test]
    fn retrive_py_instance() {
        let datapath = Path::new("../../data/Data6.mat");
        let instance = retrieve_py_data(datapath).expect("couldn't retrieve python instance");
        let instance_ref = instance.as_ref();

        Python::with_gil(|py| {
            let raw_data = instance_ref
                .getattr(py, "raw_data")
                .expect("Missing 'raw_data'");
            let raw_data_dict = raw_data
                .downcast_bound::<PyDict>(py)
                .expect("raw_data is not a dict");

            let ch1 = raw_data_dict
                .get_item("Ch1")
                .expect("Missing 'Ch1' in raw_data");

            let title = ch1
                .get_item("title")
                .expect("Missing 'title' in 'Ch1'")
                .extract::<&str>()
                .expect("title not a string");

            assert_eq!(title, "voltage")
        })
    }
}
